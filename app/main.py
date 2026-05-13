from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .artwork_catalog import build_artwork_catalog
from .config import get_settings
from .rag_service import RagService
from .schemas import (
    ArtworkImageCatalogResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    IngestResponse,
    MobileQuestionRequest,
    MobileQuestionResponse,
)

settings = get_settings()
rag_service = RagService(settings)
artworks_dir = (Path(__file__).resolve().parent.parent / "assets" / "artworks").resolve()
book_figures_dir = (Path(__file__).resolve().parent.parent / "assets" / "book_figures").resolve()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("muserag.api")

app = FastAPI(title="MuseRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/artworks", StaticFiles(directory=str(artworks_dir)), name="artworks")
book_figures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media/book_figures", StaticFiles(directory=str(book_figures_dir)), name="book_figures")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "collection": settings.muserag_collection}


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> str:
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuseRAG Chat</title>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: #fffaf1;
      --text: #2c241c;
      --muted: #6d6256;
      --accent: #a64b2a;
      --accent-dark: #7e3419;
      --border: #dccdb8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(166, 75, 42, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f0e5 0%, var(--bg) 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.2rem);
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 720px;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 96%, white 4%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(54, 38, 22, 0.08);
    }
    label {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 6px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      background: #fffdf8;
      color: var(--text);
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin: 14px 0;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: #fff7f0;
    }
    button.secondary {
      background: #e8dcc9;
      color: var(--text);
    }
    button:hover {
      background: var(--accent-dark);
    }
    button.secondary:hover {
      background: #d9c8ae;
    }
    .status {
      margin-top: 14px;
      color: var(--muted);
      min-height: 24px;
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 1.02rem;
      min-height: 120px;
    }
    .source {
      border-top: 1px solid var(--border);
      padding-top: 14px;
      margin-top: 14px;
    }
    .source:first-child {
      border-top: 0;
      padding-top: 0;
      margin-top: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 8px;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 860px) {
      .grid, .row {
        grid-template-columns: 1fr;
      }
      .wrap {
        padding: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>MuseRAG Chat</h1>
      <p>Pregunta sobre el museo y responde en base a la informacion vectorizada en Chroma, usando los documentos ya indexados en este proyecto.</p>
    </section>

    <section class="grid">
      <div class="panel">
        <label for="question">Pregunta</label>
        <textarea id="question" placeholder="Ejemplo: Que representa el Senor de Sipan y por que es importante?"></textarea>

        <div class="row">
          <div>
            <label for="room">Sala</label>
            <input id="room" placeholder="SALA_1">
          </div>
          <div>
            <label for="artwork">Obra</label>
            <input id="artwork" placeholder="obra-01">
          </div>
          <div>
            <label for="topk">Top K</label>
            <input id="topk" type="number" min="1" value="4">
          </div>
        </div>

        <div class="actions">
          <button id="askBtn">Preguntar</button>
          <button id="rebuildBtn" class="secondary">Reconstruir indice</button>
        </div>
        <div id="status" class="status">Listo para consultar.</div>
      </div>

      <div class="panel">
        <h2>Respuesta</h2>
        <div id="answer" class="answer empty">La respuesta aparecera aqui.</div>
      </div>
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Fuentes recuperadas</h2>
      <div id="sources" class="empty">Todavia no hay resultados.</div>
    </section>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");
    const askBtn = document.getElementById("askBtn");
    const rebuildBtn = document.getElementById("rebuildBtn");

    function setBusy(busy, message) {
      askBtn.disabled = busy;
      rebuildBtn.disabled = busy;
      statusEl.textContent = message;
    }

    function renderSources(sources) {
      if (!sources.length) {
        sourcesEl.className = "empty";
        sourcesEl.textContent = "No se recuperaron fuentes.";
        return;
      }

      sourcesEl.className = "";
      sourcesEl.innerHTML = sources.map((source, index) => {
        const page = source.metadata && source.metadata.page ? ` | pagina ${source.metadata.page}` : "";
        const figureRef = source.metadata && source.metadata.figure_ref ? ` | ${source.metadata.figure_ref}` : "";
        const imageHtml = source.image_url ? `<div style="margin-top: 12px;"><img src="${source.image_url}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);"></div>` : "";
        return `
          <article class="source">
            <div class="meta">[${index + 1}] ${source.kind} | score=${source.score.toFixed(3)}${page}${figureRef}</div>
            <div class="meta">${source.source}</div>
            <div>${source.text}</div>
            ${imageHtml}
          </article>
        `;
      }).join("");
    }

    async function askQuestion() {
      const question = document.getElementById("question").value.trim();
      if (!question) {
        statusEl.textContent = "Escribe una pregunta antes de consultar.";
        return;
      }

      const payload = {
        question,
        room_id: document.getElementById("room").value.trim() || null,
        artwork_id: document.getElementById("artwork").value.trim() || null,
        top_k: Number(document.getElementById("topk").value || 4)
      };

      setBusy(true, "Consultando la base vectorial...");
      try {
        const response = await fetch("/chat/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Error inesperado al consultar.");
        }
        const data = await response.json();
        answerEl.className = "answer";
        answerEl.textContent = data.answer;
        renderSources(data.sources || []);
        setBusy(false, "Consulta completada.");
      } catch (error) {
        answerEl.className = "answer empty";
        answerEl.textContent = "No se pudo generar la respuesta.";
        renderSources([]);
        setBusy(false, error.message);
      }
    }

    async function rebuildIndex() {
      setBusy(true, "Reconstruyendo indice...");
      try {
        const response = await fetch("/ingest/rebuild", { method: "POST" });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Error inesperado al reconstruir.");
        }
        const data = await response.json();
        setBusy(false, `Indice reconstruido: ${data.indexed_documents} fragmentos.`);
      } catch (error) {
        setBusy(false, error.message);
      }
    }

    askBtn.addEventListener("click", askQuestion);
    rebuildBtn.addEventListener("click", rebuildIndex);
  </script>
</body>
</html>
    """


@app.get("/catalog/artworks", response_model=ArtworkImageCatalogResponse)
def artwork_catalog() -> ArtworkImageCatalogResponse:
    return build_artwork_catalog(artworks_dir)


@app.post("/ingest/rebuild", response_model=IngestResponse)
def rebuild_ingest() -> IngestResponse:
    try:
        indexed = rag_service.rebuild_index()
        return IngestResponse(indexed_documents=indexed, collection=settings.muserag_collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/query", response_model=ChatQueryResponse)
def chat_query(payload: ChatQueryRequest) -> ChatQueryResponse:
    try:
        answer, sources, meta = rag_service.answer_question(payload)
        return ChatQueryResponse(
            answer=answer,
            sources=sources,
            used_artwork_context=payload.artwork_context is not None,
            meta=meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/preguntar", response_model=MobileQuestionResponse)
def mobile_question(payload: MobileQuestionRequest) -> MobileQuestionResponse:
    started_at = time.perf_counter()
    try:
        if settings.muserag_log_interactions:
            logger.info(
                "Solicitud app | pregunta=%r | museo=%s | sala=%s | obra=%s",
                payload.pregunta,
                payload.museo or "N/D",
                payload.sala or "N/D",
                payload.obra or "N/D",
            )

        internal_payload = ChatQueryRequest(
            question=payload.pregunta,
            museum_id=payload.museo,
            room_id=payload.sala,
            artwork_id=payload.obra,
            artwork_context=payload.artwork_context,
        )
        answer, sources, meta = rag_service.answer_question(internal_payload)
        if settings.muserag_log_interactions:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Respuesta app | obra=%s | fuentes=%s | duracion_ms=%.1f | respuesta=%r",
                payload.obra or "N/D",
                len(sources),
                elapsed_ms,
                answer[:300],
            )
        return MobileQuestionResponse(
            respuesta=answer,
            fuentes=sources,
            museo=payload.museo,
            sala=payload.sala,
            obra=payload.obra,
            meta=meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
