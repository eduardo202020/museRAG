from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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
    reader = PdfReader(str(pdf_path))
    chunks: list[DocumentChunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            continue
        for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
            chunks.append(
                DocumentChunk(
                    id=f"pdf-page-{page_index}-chunk-{chunk_index}",
                    text=chunk,
                    metadata={
                        "source": str(pdf_path),
                        "kind": "pdf",
                        "page": page_index,
                        "chunk": chunk_index,
                    },
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
