from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArtworkContext(BaseModel):
    id: str | None = None
    title: str | None = None
    author: str | None = None
    year: str | None = None
    period: str | None = None
    technique: str | None = None
    summary: str | None = None
    context: str | None = None
    room_relation: str | None = None
    location_hint: str | None = None
    suggested_questions: list[str] = Field(default_factory=list)


class ChatQueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str
    museum_id: str | None = Field(default=None, alias="museo")
    room_id: str | None = None
    artwork_id: str | None = None
    session_id: str | None = None
    top_k: int | None = None
    artwork_context: ArtworkContext | None = None


class SourceSnippet(BaseModel):
    id: str
    source: str
    kind: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_url: str | None = None
    source_label: str | None = None


class ResponseMeta(BaseModel):
    total_ms: float
    retrieval_ms: float
    generation_ms: float
    source_count: int
    support_level: str | None = None
    applied_filters: list[str] = Field(default_factory=list)


class ChatQueryResponse(BaseModel):
    answer: str
    markdown: str | None = None
    sources: list[SourceSnippet]
    used_artwork_context: bool
    meta: ResponseMeta | None = None


class MobileQuestionRequest(BaseModel):
    pregunta: str
    museo: str | None = None
    sala: str | None = None
    obra: str | None = None
    session_id: str | None = None
    artwork_context: ArtworkContext | None = None


class MobileQuestionResponse(BaseModel):
    respuesta: str
    markdown: str | None = None
    fuentes: list[SourceSnippet]
    museo: str | None = None
    sala: str | None = None
    obra: str | None = None
    session_id: str | None = None
    meta: ResponseMeta | None = None


class IngestResponse(BaseModel):
    indexed_documents: int
    collection: str


class ArtworkImageItem(BaseModel):
    filename: str
    room: str
    relative_path: str
    url: str


class ArtworkImageRoom(BaseModel):
    room: str
    total: int
    items: list[ArtworkImageItem]


class ArtworkImageCatalogResponse(BaseModel):
    total_rooms: int
    total_images: int
    rooms: list[ArtworkImageRoom]
