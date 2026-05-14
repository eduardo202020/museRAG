from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Any

import chromadb
from openai import OpenAI

from .config import Settings
from .loaders import (
    DocumentChunk,
    load_artwork_chunks_from_ts,
    load_museum_json_chunks,
    load_pdf_chunks,
    load_text_file_chunks,
)
from .schemas import ArtworkContext, ChatQueryRequest, ResponseMeta, SourceSnippet

logger = logging.getLogger("muserag.rag")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
STRUCTURED_RESPONSE_TEMPLATE = (
    "Devuelve la respuesta final con este formato exacto y en este orden:\n"
    "Respuesta breve: <explicacion principal en 2 o 3 frases>\n"
    "Dato clave: <un hallazgo, relacion o detalle puntual en 1 o 2 frases>\n"
    "Imagen relacionada: <explica brevemente que imagen o tipo de imagen acompana la respuesta; "
    "si no hay imagen util en las fuentes, di 'No encontre una imagen relacionada clara en las fuentes recuperadas.'>\n"
    "Siguiente mirada: <sugiere con iniciativa de guia experimentado un detalle para observar, comparar o preguntar despues>\n"
    "No uses vietas, markdown ni titulos alternativos."
)
MAX_SESSION_TURNS = 3
MAX_ACTIVE_SESSIONS = 100
MIN_CONTEXTUAL_RESULTS = 2
LOW_SUPPORT_THRESHOLD = 0.22


class SessionMemoryTurn(dict[str, str]):
    pass


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(base_url=settings.lm_studio_base_url, api_key="lm-studio")
        self.chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.chroma.get_or_create_collection(name=settings.muserag_collection, metadata={"hnsw:space": "cosine"})
        self.session_memories: dict[str, deque[SessionMemoryTurn]] = {}

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
        documents.extend(load_artwork_chunks_from_ts(self.settings.app_data_ts_path))
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

    @staticmethod
    def _source_label_for(kind: str) -> str:
        labels = {
            "pdf": "Libro del museo",
            "museum_json": "Narrativa de sala",
            "app_artwork": "Ficha curatorial",
            "app_data_ts": "Dataset curatorial",
        }
        return labels.get(kind, "Fuente del museo")

    def _run_query(
        self,
        question: str,
        top_k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[SourceSnippet]:
        query_embedding = self._embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        snippets: list[SourceSnippet] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            image_url = metadata.get("image_url")
            kind = str(metadata.get("kind", "unknown"))
            snippets.append(
                SourceSnippet(
                    id=str(ids[idx]),
                    source=str(metadata.get("source", "unknown")),
                    kind=kind,
                    score=max(0.0, 1.0 - distance),
                    text=text,
                    metadata={str(k): v for k, v in metadata.items()},
                    image_url=str(image_url) if image_url else None,
                    source_label=self._source_label_for(kind),
                )
            )
        return snippets

    @staticmethod
    def _dedupe_sources(sources: list[SourceSnippet]) -> list[SourceSnippet]:
        deduped: list[SourceSnippet] = []
        seen_ids: set[str] = set()
        for source in sorted(sources, key=lambda item: item.score, reverse=True):
            if source.id in seen_ids:
                continue
            seen_ids.add(source.id)
            deduped.append(source)
        return deduped

    def _query_sources(self, payload: ChatQueryRequest, top_k: int) -> tuple[list[SourceSnippet], list[str]]:
        collected: list[SourceSnippet] = []
        applied_filters: list[str] = []

        if payload.artwork_id:
            artwork_sources = self._run_query(
                payload.question,
                top_k=max(top_k, MIN_CONTEXTUAL_RESULTS),
                where={"artwork_id": payload.artwork_id},
            )
            if artwork_sources:
                applied_filters.append(f"artwork_id={payload.artwork_id}")
                collected.extend(artwork_sources)

        if payload.room_id and len(self._dedupe_sources(collected)) < MIN_CONTEXTUAL_RESULTS:
            room_sources = self._run_query(
                payload.question,
                top_k=max(top_k, MIN_CONTEXTUAL_RESULTS),
                where={"room_id": payload.room_id},
            )
            if room_sources:
                applied_filters.append(f"room_id={payload.room_id}")
                collected.extend(room_sources)

        collected.extend(self._run_query(payload.question, top_k=top_k))
        return self._dedupe_sources(collected)[:top_k], applied_filters

    def _trim_source_text(self, text: str) -> str:
        max_chars = max(120, self.settings.muserag_max_source_chars)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

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

    def _get_session_memory_block(self, session_id: str | None) -> str:
        if not session_id:
            return ""

        turns = self.session_memories.get(session_id)
        if not turns:
            return ""

        blocks = []
        for index, turn in enumerate(turns, start=1):
            blocks.append(
                f"Turno previo {index}:\n"
                f"Pregunta: {turn['question']}\n"
                f"Respuesta: {turn['answer']}"
            )
        return "\n\n".join(blocks)

    def _remember_turn(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return

        if session_id not in self.session_memories and len(self.session_memories) >= MAX_ACTIVE_SESSIONS:
            oldest_session_id = next(iter(self.session_memories))
            self.session_memories.pop(oldest_session_id, None)

        session_turns = self.session_memories.setdefault(
            session_id,
            deque(maxlen=MAX_SESSION_TURNS),
        )
        session_turns.append(
            SessionMemoryTurn(question=question.strip(), answer=answer.strip())
        )

    @staticmethod
    def _compute_support_level(sources: list[SourceSnippet], artwork_context: ArtworkContext | None) -> str:
        if not sources:
            return "bajo"

        top_score = sources[0].score
        if top_score >= 0.55 or (top_score >= 0.35 and artwork_context is not None):
            return "alto"
        if top_score >= LOW_SUPPORT_THRESHOLD or artwork_context is not None:
            return "medio"
        return "bajo"

    @staticmethod
    def _build_low_context_answer(artwork_context: ArtworkContext | None) -> str:
        if artwork_context and artwork_context.title:
            return (
                f"Respuesta breve: Puedo orientarte sobre {artwork_context.title}, pero ahora mismo el sustento recuperado es limitado y prefiero no afirmar mas de lo que muestran las fuentes.\n"
                f"Dato clave: Con el contexto curatorial disponible, esta obra se relaciona con {artwork_context.room_relation or 'la narrativa de su sala'} y conviene hacer una pregunta mas puntual para profundizar.\n"
                "Imagen relacionada: No encontre una imagen relacionada clara en las fuentes recuperadas.\n"
                f"Siguiente mirada: Si quieres, puedo ayudarte mejor si me preguntas por un detalle visible de {artwork_context.title}, su tecnica o su relacion con la sala."
            )

        return (
            "Respuesta breve: En este momento no encontre suficiente contexto confiable para responder con precision a esa pregunta.\n"
            "Dato clave: Si quieres, prueba con una pregunta mas concreta sobre la obra actual, la sala o su importancia dentro del recorrido.\n"
            "Imagen relacionada: No encontre una imagen relacionada clara en las fuentes recuperadas.\n"
            "Siguiente mirada: Podemos seguir por una pista mas especifica, como el material de la obra, el personaje representado o lo que comunica dentro de la sala."
        )

    def _build_messages(self, payload: ChatQueryRequest, sources: list[SourceSnippet]) -> list[dict[str, Any]]:
        source_text = "\n\n".join(
            [
                f"Fuente {index + 1} ({source.source_label or source.kind}, score={source.score:.3f}):\n{self._trim_source_text(source.text)}"
                for index, source in enumerate(sources)
            ]
        )
        artwork_context = self._artwork_context_block(payload.artwork_context)
        session_memory = self._get_session_memory_block(payload.session_id)
        support_level = self._compute_support_level(sources, payload.artwork_context)

        system_prompt = (
            "Eres MuseIQ, un guia de museo en espanol. "
            "Debes responder exclusivamente en espanol latinoamericano. "
            "No uses chino, ingles ni otro idioma salvo nombres propios o terminos arqueologicos inevitables. "
            "Responde con claridad, tono cercano y precision historica. "
            "Usa primero el contexto de la obra actual si existe y luego el conocimiento recuperado. "
            "Si existe memoria conversacional reciente, usala para mantener continuidad sin repetir toda la respuesta anterior. "
            "Actua como un guia experimentado: no te limites a contestar, tambien sugiere un siguiente detalle para mirar o una relacion valiosa para seguir explorando. "
            "Si la respuesta no esta sustentada por el contexto, dilo con honestidad y evita inventar datos. "
            "Responde en maximo 6 frases. "
            f"{STRUCTURED_RESPONSE_TEMPLATE}"
        )
        user_prompt = (
            f"Pregunta del visitante: {payload.question}\n\n"
            f"Contexto actual de la app:\n"
            f"- museum_id: {payload.museum_id or 'N/D'}\n"
            f"- room_id: {payload.room_id or 'N/D'}\n"
            f"- artwork_id: {payload.artwork_id or 'N/D'}\n"
            f"- support_level: {support_level}\n\n"
            f"Memoria conversacional reciente:\n{session_memory or 'No disponible'}\n\n"
            f"Contexto de obra actual:\n{artwork_context or 'No disponible'}\n\n"
            f"Fragmentos recuperados:\n{source_text or 'No hay fragmentos recuperados.'}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(CJK_PATTERN.search(text))

    def _regenerate_in_spanish(self, messages: list[dict[str, Any]], answer: str) -> str:
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Corrige la respuesta anterior. "
                    "Devuelve una nueva version exclusivamente en espanol latinoamericano. "
                    "Elimina cualquier caracter o frase en chino u otro idioma. "
                    "Conserva el sentido historico y responde en maximo 6 frases. "
                    f"{STRUCTURED_RESPONSE_TEMPLATE}"
                ),
            },
            *messages,
            {
                "role": "assistant",
                "content": answer,
            },
            {
                "role": "user",
                "content": "Reescribe esa respuesta solo en espanol claro para un visitante del museo.",
            },
        ]
        retry_response = self.client.chat.completions.create(
            model=self.settings.lm_studio_chat_model,
            temperature=0.1,
            messages=retry_messages,
            max_tokens=self.settings.muserag_chat_max_tokens,
        )
        retried_answer = retry_response.choices[0].message.content or ""
        return retried_answer.strip()

    def answer_question(self, payload: ChatQueryRequest) -> tuple[str, list[SourceSnippet], ResponseMeta]:
        top_k = payload.top_k or self.settings.muserag_top_k
        started_at = time.perf_counter()

        embed_and_query_started_at = time.perf_counter()
        sources, applied_filters = self._query_sources(payload, top_k=top_k)
        retrieval_ms = (time.perf_counter() - embed_and_query_started_at) * 1000
        support_level = self._compute_support_level(sources, payload.artwork_context)

        if support_level == "bajo" and payload.artwork_context is None:
            answer = self._build_low_context_answer(payload.artwork_context)
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Consulta RAG | top_k=%s | soporte=%s | filtros=%s | fuentes=%s | retrieval_ms=%.1f | generation_ms=0.0 | total_ms=%.1f",
                top_k,
                support_level,
                applied_filters,
                len(sources),
                retrieval_ms,
                total_ms,
            )
            return (
                answer,
                sources,
                ResponseMeta(
                    total_ms=round(total_ms, 1),
                    retrieval_ms=round(retrieval_ms, 1),
                    generation_ms=0.0,
                    source_count=len(sources),
                    support_level=support_level,
                    applied_filters=applied_filters,
                ),
            )

        messages = self._build_messages(payload, sources)
        generation_started_at = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.lm_studio_chat_model,
            temperature=0.2,
            messages=messages,
            max_tokens=self.settings.muserag_chat_max_tokens,
        )
        generation_ms = (time.perf_counter() - generation_started_at) * 1000
        total_ms = (time.perf_counter() - started_at) * 1000

        answer = response.choices[0].message.content or "No pude generar una respuesta en este momento."
        answer = answer.strip()
        if self._contains_cjk(answer):
            logger.warning("Se detecto salida con caracteres CJK; reintentando respuesta en espanol.")
            retried_answer = self._regenerate_in_spanish(messages, answer)
            if retried_answer and not self._contains_cjk(retried_answer):
                answer = retried_answer
            else:
                answer = (
                    "Puedo ayudarte con esa pregunta, pero en este momento hubo un problema de idioma en la generacion. "
                    "Intenta formularla otra vez para responderte en espanol."
                )
        self._remember_turn(payload.session_id, payload.question, answer)
        logger.info(
            "Consulta RAG | top_k=%s | soporte=%s | filtros=%s | fuentes=%s | retrieval_ms=%.1f | generation_ms=%.1f | total_ms=%.1f",
            top_k,
            support_level,
            applied_filters,
            len(sources),
            retrieval_ms,
            generation_ms,
            total_ms,
        )
        return (
            answer,
            sources,
            ResponseMeta(
                total_ms=round(total_ms, 1),
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
                source_count=len(sources),
                support_level=support_level,
                applied_filters=applied_filters,
            ),
        )
