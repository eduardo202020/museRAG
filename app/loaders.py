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
        print("[WARN] pdf2image no disponible. Intenta: pip install pdf2image")
        print("   Asegurate de tener poppler instalado en tu sistema.")
        return {}

    print(f"[INFO] Extrayendo imagenes del PDF: {pdf_path.name}")
    
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
                    print(f"   [OK] Pagina {page_index}: guardada {filename}")
                else:
                    # Page without explicit figure reference
                    filename = f"page_{page_index}.png"
                    filepath = output_dir / filename
                    page_image.save(str(filepath), "PNG")
                    print(f"   [INFO] Pagina {page_index}: guardada {filename} (sin referencia de figura)")

            except Exception as e:
                print(f"   [ERROR] Pagina {page_index}: {type(e).__name__}")

        print(f"[OK] Extraccion completada: {len(image_map)} imagenes con referencias de figuras\n")
        return image_map

    except Exception as e:
        print(f"[ERROR] Error al convertir PDF: {e}")
        print("   Asegurate de tener poppler instalado y visible en PATH.")
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


def _extract_balanced_blocks(raw_text: str, *, anchor: str) -> list[str]:
    anchor_index = raw_text.find(anchor)
    if anchor_index < 0:
        return []

    list_start = raw_text.find("[", anchor_index)
    if list_start < 0:
        return []

    depth = 0
    current_start: int | None = None
    blocks: list[str] = []

    for index in range(list_start, len(raw_text)):
        char = raw_text[index]

        if char == "{":
            depth += 1
            if depth == 1:
                current_start = index
        elif char == "}":
            if depth == 1 and current_start is not None:
                blocks.append(raw_text[current_start : index + 1])
                current_start = None
            depth = max(0, depth - 1)
        elif char == "]" and depth == 0:
            break

    return blocks


def _find_artworks_list_start(raw_text: str) -> int:
    museum_anchor = raw_text.find("export const museumMock")
    if museum_anchor < 0:
        return -1

    artworks_anchor = raw_text.find("artworks:", museum_anchor)
    if artworks_anchor < 0:
        return -1

    return raw_text.find("[", artworks_anchor)


def _extract_string_field(block: str, field_name: str) -> str | None:
    pattern = rf'{field_name}:\s*"([^"]+)"'
    match = re.search(pattern, block)
    return match.group(1).strip() if match else None


def _extract_number_field(block: str, field_name: str) -> int | None:
    pattern = rf"{field_name}:\s*(\d+)"
    match = re.search(pattern, block)
    return int(match.group(1)) if match else None


def _extract_array_field(block: str, field_name: str) -> list[str]:
    pattern = rf"{field_name}:\s*\[(.*?)\]"
    match = re.search(pattern, block, re.DOTALL)
    if not match:
        return []

    return [value.strip() for value in re.findall(r'"([^"]+)"', match.group(1))]


def load_artwork_chunks_from_ts(file_path: Path) -> list[DocumentChunk]:
    raw_text = file_path.read_text(encoding="utf-8")
    list_start = _find_artworks_list_start(raw_text)
    if list_start < 0:
        return []

    blocks = _extract_balanced_blocks(raw_text[list_start:], anchor="[")
    chunks: list[DocumentChunk] = []

    for block in blocks:
        artwork_id = _extract_string_field(block, "id")
        room_id = _extract_string_field(block, "roomId")
        title = _extract_string_field(block, "title")

        if not artwork_id or not room_id or not title:
            continue

        author = _extract_string_field(block, "author") or "N/D"
        year = _extract_string_field(block, "year") or "N/D"
        period = _extract_string_field(block, "period") or "N/D"
        technique = _extract_string_field(block, "technique") or "N/D"
        summary = _extract_string_field(block, "summary") or "N/D"
        context = _extract_string_field(block, "context") or "N/D"
        room_relation = _extract_string_field(block, "roomRelation") or "N/D"
        location_hint = _extract_string_field(block, "locationHint") or "N/D"
        zone = _extract_string_field(block, "zone") or "N/D"
        image = _extract_string_field(block, "image")
        order = _extract_number_field(block, "order") or 0
        tags = _extract_array_field(block, "tags")
        suggested_questions = _extract_array_field(block, "suggestedQuestions")

        lines = [
            f"Obra: {title}",
            f"ID de obra: {artwork_id}",
            f"Sala: {room_id}",
            f"Zona: {zone}",
            f"Autor: {author}",
            f"Anio: {year}",
            f"Periodo: {period}",
            f"Tecnica: {technique}",
            f"Resumen: {summary}",
            f"Contexto: {context}",
            f"Relacion con la sala: {room_relation}",
            f"Ubicacion sugerida: {location_hint}",
        ]
        if tags:
            lines.append(f"Etiquetas: {', '.join(tags)}")
        if suggested_questions:
            lines.append(f"Preguntas sugeridas: {', '.join(suggested_questions)}")

        metadata: dict[str, str | int] = {
            "source": str(file_path),
            "kind": "app_artwork",
            "room_id": room_id,
            "artwork_id": artwork_id,
            "artwork_title": title,
            "zone": zone,
            "order": order,
        }
        if image:
            metadata["image_path"] = image
            relative_image = image.replace("artworks/", "", 1)
            metadata["image_url"] = f"/media/artworks/{relative_image}"

        chunks.append(
            DocumentChunk(
                id=f"app-artwork-{artwork_id}",
                text="\n".join(lines),
                metadata=metadata,
            )
        )

    return chunks
