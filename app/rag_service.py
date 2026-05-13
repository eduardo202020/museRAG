from __future__ import annotations

from typing import Any

import chromadb
from openai import OpenAI

from .config import Settings
from .loaders import DocumentChunk, load_museum_json_chunks, load_pdf_chunks, load_text_file_chunks
from .schemas import ArtworkContext, ChatQueryRequest, SourceSnippet


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(base_url=settings.lm_studio_base_url, api_key="lm-studio")
        self.chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.chroma.get_or_create_collection(name=settings.muserag_collection, metadata={"hnsw:space": "cosine"})

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.settings.lm_studio_embed_model, input=texts)
        return [item.embedding for item in response.data]

    def rebuild_index(self) -> int:
        try:
            self.chroma.delete_collection(self.settings.muserag_collection)
        except Exception:
            pass
        self.collection = self.chroma.get_or_create_collection(name=self.settings.muserag_collection, metadata={"hnsw:space": "cosine"})

        documents: list[DocumentChunk] = []
        documents.extend(load_pdf_chunks(self.settings.pdf_path))
        documents.extend(load_museum_json_chunks(self.settings.museum_json_path))
        documents.extend(load_text_file_chunks(self.settings.app_data_ts_path, kind="app_data_ts"))

        if not documents:
            return 0

        embeddings = self._embed_texts([doc.text for doc in documents])
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
        )
        return len(documents)

    def count_documents(self) -> int:
        return self.collection.count()

    def _query_sources(self, question: str, top_k: int) -> list[SourceSnippet]:
        query_embedding = self._embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        snippets: list[SourceSnippet] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            snippets.append(
                SourceSnippet(
                    id=str(ids[idx]),
                    source=str(metadata.get("source", "unknown")),
                    kind=str(metadata.get("kind", "unknown")),
                    score=max(0.0, 1.0 - distance),
                    text=text,
                    metadata={str(k): v for k, v in metadata.items()},
                )
            )
        return snippets

    @staticmethod
    def _artwork_context_block(artwork_context: ArtworkContext | None) -> str:
        if not artwork_context:
            return ""

        lines = [
            f"ID de obra: {artwork_context.id or 'N/D'}",
            f"Titulo: {artwork_context.title or 'N/D'}",
            f"Autor: {artwork_context.author or 'N/D'}",
            f"Anio: {artwork_context.year or 'N/D'}",
            f"Periodo: {artwork_context.period or 'N/D'}",
            f"Tecnica: {artwork_context.technique or 'N/D'}",
            f"Resumen: {artwork_context.summary or 'N/D'}",
            f"Contexto: {artwork_context.context or 'N/D'}",
            f"Relacion con la sala: {artwork_context.room_relation or 'N/D'}",
            f"Ubicacion sugerida: {artwork_context.location_hint or 'N/D'}",
        ]
        if artwork_context.suggested_questions:
            lines.append(f"Preguntas sugeridas: {', '.join(artwork_context.suggested_questions)}")
        return "\n".join(lines)

    def _build_messages(self, payload: ChatQueryRequest, sources: list[SourceSnippet]) -> list[dict[str, Any]]:
        source_text = "\n\n".join(
            [
                f"Fuente {index + 1} ({source.kind}, score={source.score:.3f}):\n{source.text}"
                for index, source in enumerate(sources)
            ]
        )
        artwork_context = self._artwork_context_block(payload.artwork_context)

        system_prompt = (
            "Eres MuseIQ, un guia de museo en espanol. "
            "Responde con claridad, tono cercano y precision historica. "
            "Usa primero el contexto de la obra actual si existe y luego el conocimiento recuperado. "
            "Si la respuesta no esta sustentada por el contexto, dilo con honestidad y evita inventar datos. "
            "Responde en maximo 6 frases."
        )
        user_prompt = (
            f"Pregunta del visitante: {payload.question}\n\n"
            f"Contexto actual de la app:\n"
            f"- museum_id: {payload.museum_id or 'N/D'}\n"
            f"- room_id: {payload.room_id or 'N/D'}\n"
            f"- artwork_id: {payload.artwork_id or 'N/D'}\n\n"
            f"Contexto de obra actual:\n{artwork_context or 'No disponible'}\n\n"
            f"Fragmentos recuperados:\n{source_text or 'No hay fragmentos recuperados.'}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def answer_question(self, payload: ChatQueryRequest) -> tuple[str, list[SourceSnippet]]:
        top_k = payload.top_k or self.settings.muserag_top_k
        sources = self._query_sources(payload.question, top_k=top_k)
        messages = self._build_messages(payload, sources)
        response = self.client.chat.completions.create(
            model=self.settings.lm_studio_chat_model,
            temperature=0.2,
            messages=messages,
        )
        answer = response.choices[0].message.content or "No pude generar una respuesta en este momento."
        return answer.strip(), sources
