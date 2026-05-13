from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
from PIL import Image
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_figure_references(text: str) -> list[tuple[str, int]]:
    """
    Extract figure references like 'Fig. XX', 'Figura XX' with their positions.
    Returns list of (figure_number, position) tuples sorted by position.
    """
    pattern = r"(?:Fig|Figura|Figure)\.?\s*(\d{1,3})"
    matches = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        fig_num = match.group(1)
        position = match.start()
        matches.append((fig_num, position))
    return sorted(set(matches), key=lambda x: x[1])


def get_figure_in_chunk(chunk_text: str) -> str | None:
    """Extract the first figure reference in a chunk, if any."""
    refs = extract_figure_references(chunk_text)
    return refs[0][0] if refs else None


def find_figure_image(figure_num: str, figures_dir: Path) -> str | None:
    """
    Find an image file for a figure number in the figures directory.
    Returns relative path if found, None otherwise.
    """
    if not figures_dir.exists():
        return None
    
    # Look for files like Fig_01_*.png, Fig_1_*.png, etc.
    for pattern in [f"Fig_{figure_num}_*.png", f"Fig_{figure_num.lstrip('0')}_*.png"]:
        matches = list(figures_dir.glob(pattern))
        if matches:
            return matches[0].name
    
    return None


def extract_pdf_images_batch(pdf_path: Path, output_dir: Path) -> dict[str, str]:
    """
    Extract images from PDF pages using pdf2image or fallback to pypdf.
    Returns mapping of figure_number -> image_filename.
    
    Handles JPEG2000 and other complex formats via pdf2image (poppler).
    Falls back to direct XObject extraction for simpler PDFs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, str] = {}

    if not HAS_PDF2IMAGE:
        print("⚠️  pdf2image no disponible. Intenta: pip install pdf2image")
        print("   Asegúrate de tener poppler instalado en tu sistema.")
        return {}

    print(f"🖼️ Extrayendo imágenes del PDF: {pdf_path.name}")
    
    try:
        # Convert PDF to images using pdf2image (handles JPEG2000, etc.)
        images = convert_from_path(str(pdf_path), dpi=150)
        
        # Get figure references from text
        reader = PdfReader(str(pdf_path))
        page_figure_refs: dict[int, list[str]] = {}
        
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            if page_text:
                refs = extract_figure_references(page_text)
                page_figure_refs[page_index] = [ref[0] for ref in refs]
        
        # Process images
        for page_index, page_image in enumerate(images, start=1):
            try:
                # Get figure references for this page
                fig_refs = page_figure_refs.get(page_index, [])
                
                if fig_refs:
                    # One image per page with figure reference
                    fig_num = fig_refs[0]
                    filename = f"Fig_{fig_num}_p{page_index}.png"
                    filepath = output_dir / filename
                    page_image.save(str(filepath), "PNG")
                    image_map[fig_num] = filename
                    print(f"   ✓ Página {page_index}: guardada {filename}")
                else:
                    # Page without explicit figure reference
                    filename = f"page_{page_index}.png"
                    filepath = output_dir / filename
                    page_image.save(str(filepath), "PNG")
                    print(f"   ℹ️  Página {page_index}: guardada {filename} (sin referencia de figura)")

            except Exception as e:
                print(f"   ✗ Página {page_index}: {type(e).__name__}")

        print(f"✅ Extracción completada: {len(image_map)} imágenes con referencias de figuras\n")
        return image_map

    except Exception as e:
        print(f"❌ Error al convertir PDF: {e}")
        print(f"   Asegúrate de tener poppler instalado (apt-get install poppler-utils)")
        return {}


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf_chunks(pdf_path: Path) -> list[DocumentChunk]:
    """
    Load PDF chunks and link to extracted figures by page.
    Images must be pre-extracted using extract_images.py script.
    Links first available figure from each page to all chunks from that page.
    """
    reader = PdfReader(str(pdf_path))
    chunks: list[DocumentChunk] = []
    
    # Look for pre-extracted figures in assets/book_figures/
    figures_dir = pdf_path.parent / "assets" / "book_figures"
    
    # Build page -> image mapping
    page_to_images: dict[int, list[tuple[str, str]]] = {}  # page -> [(fig_num, filename)]
    
    if figures_dir.exists():
        image_files = list(figures_dir.glob("*.png"))
        for img_file in image_files:
            filename = img_file.name
            page_num = None
            fig_num = "?"
            
            # Parse filename: Fig_XX_pYY.png or page_YY.png
            if filename.startswith("Fig_"):
                # Format: Fig_1_p15.png, Fig_123_p45.png
                parts = filename.replace(".png", "").split("_p")
                if len(parts) == 2:
                    fig_part = parts[0].replace("Fig_", "")  # "1", "123", etc.
                    page_part = parts[1]  # "15", "45", etc.
                    try:
                        page_num = int(page_part)
                        fig_num = fig_part
                    except ValueError:
                        pass
            elif filename.startswith("page_"):
                # Format: page_15.png
                page_part = filename.replace("page_", "").replace(".png", "")
                try:
                    page_num = int(page_part)
                    fig_num = "page"
                except ValueError:
                    pass
            
            if page_num and page_num not in page_to_images:
                page_to_images[page_num] = []
            if page_num:
                page_to_images[page_num].append((fig_num, filename))

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            continue
        
        # Get images for this page
        page_images = page_to_images.get(page_index, [])
        first_image_url = None
        first_image_ref = None
        
        if page_images:
            # Use first available image from this page
            fig_num, filename = page_images[0]
            first_image_url = f"/media/book_figures/{filename}"
            if fig_num != "page":
                first_image_ref = f"Fig. {fig_num}"
        
        for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
            metadata = {
                "source": str(pdf_path),
                "kind": "pdf",
                "page": page_index,
                "chunk": chunk_index,
            }
            
            if first_image_url:
                metadata["image_url"] = first_image_url
                if first_image_ref:
                    metadata["figure_ref"] = first_image_ref
            
            chunks.append(
                DocumentChunk(
                    id=f"pdf-page-{page_index}-chunk-{chunk_index}",
                    text=chunk,
                    metadata=metadata,
                )
            )

    return chunks


def build_room_text(room_id: str, room_payload: dict) -> str:
    lines = [
        f"Sala: {room_id}",
        f"Narracion general: {normalize_text(room_payload.get('narration', ''))}",
    ]
    for zone_key in ("Z1", "Z2", "Z3"):
        zone = room_payload.get(zone_key)
        if zone:
            lines.append(f"{zone_key}: {normalize_text(zone.get('narration', ''))}")
    return "\n".join(line for line in lines if line.strip())


def load_museum_json_chunks(museum_json_path: Path) -> list[DocumentChunk]:
    payload = json.loads(museum_json_path.read_text(encoding="utf-8"))
    chunks: list[DocumentChunk] = []

    for room_id, room_payload in payload.items():
        room_text = build_room_text(room_id, room_payload)
        if not room_text.strip():
            continue
        chunks.append(
            DocumentChunk(
                id=f"museum-room-{room_id}",
                text=room_text,
                metadata={
                    "source": str(museum_json_path),
                    "kind": "museum_json",
                    "room_id": room_id,
                },
            )
        )

    return chunks


def load_text_file_chunks(file_path: Path, *, kind: str) -> list[DocumentChunk]:
    raw_text = file_path.read_text(encoding="utf-8")
    chunks: list[DocumentChunk] = []

    for chunk_index, chunk in enumerate(chunk_text(raw_text, chunk_size=1200, overlap=180), start=1):
        chunks.append(
            DocumentChunk(
                id=f"{kind}-chunk-{chunk_index}",
                text=chunk,
                metadata={
                    "source": str(file_path),
                    "kind": kind,
                    "chunk": chunk_index,
                },
            )
        )

    return chunks
