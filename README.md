# MuseRAG

Backend RAG local para `museiqApp`, orientado a experiencias conversacionales en museo.

Su funcion principal hoy es responder preguntas contextualizadas por:

- museo
- sala
- obra
- contexto curatorial de la pieza actual

La version actual esta configurada para un museo, pero la estructura de entrada ya contempla `museo` para evolucionar luego a un escenario multi-museo sin cambiar el contrato principal del API.

Stack actual:

- `FastAPI` para la API HTTP
- `Chroma` como vector store persistente
- `LM Studio` como proveedor local OpenAI-compatible

## Proposito del proyecto

MuseRAG actua como la capa de conocimiento de `museiqApp`. Su trabajo es tomar una pregunta del visitante y enriquecerla con el contexto de navegacion de la app para producir respuestas mas utiles, naturales y situadas dentro del recorrido.

En la practica, el backend puede combinar:

- pregunta libre del visitante
- museo actual
- sala actual
- obra actual
- metadatos de la obra como titulo, autor, periodo, tecnica y resumen
- fragmentos recuperados desde la base vectorial

Esto permite una experiencia RAG enfocada en mediacion cultural, no solo en busqueda documental.

## Estado actual

El proyecto esta alineado con una maqueta curatorial de `2` salas:

- `SALA_1`: `12` obras
- `SALA_2`: `10` obras
- Total: `22` obras

La app cliente tambien fue ajustada para este escenario:

- `museiqApp/datos.ts` define las `22` obras activas
- `museiqApp/content/museum.json` contiene narrativas de `2` salas
- `museiqApp/assets/images/artworks/` guarda las `22` imagenes organizadas en:
  - `sala-01/`
  - `sala-02/`

MuseRAG tambien expone estas imagenes directamente como servidor:

- `assets/artworks/sala-01/`
- `assets/artworks/sala-02/`

## Que indexa MuseRAG

MuseRAG construye su conocimiento a partir de tres fuentes:

- `tumbas-reales-sipan.pdf`
- `../museiqApp/content/museum.json`
- `../museiqApp/datos.ts`

Con eso puede:

- responder preguntas generales sobre el recorrido
- responder con contexto de sala y zona
- usar el contexto de la obra actual si `museiqApp` lo envia
- usar metadatos curatoriales de la obra actual para enriquecer la pregunta
- devolver fragmentos fuente desde Chroma

## Modelo de consulta actual

El endpoint principal para la app movil es:

- `POST /api/preguntar`

Payload esperado:

```json
{
  "pregunta": "Quien fue el Senor de Sipan?",
  "museo": "tumbas-reales-de-sipan",
  "sala": "SALA_1",
  "obra": "Senor de Sipan",
  "artwork_context": {
    "id": "obra-1-1-L",
    "title": "Senor de Sipan",
    "author": "Elite moche",
    "year": "siglo III d.C.",
    "period": "Moche Medio",
    "technique": "Fotografia documental",
    "summary": "Retrato del hallazgo mas emblematico del norte peruano.",
    "context": "Resume la magnificencia del gobernante enterrado con metales, textiles y simbolos de autoridad.",
    "room_relation": "Introduce el relato de poder y entierro de la Sala 1.",
    "location_hint": "Entrada, lado izquierdo.",
    "suggested_questions": [
      "Quien fue el Senor de Sipan?"
    ]
  }
}
```

Este contrato ya deja preparado el camino para:

- varios museos con colecciones distintas
- respuestas mas precisas por obra
- futuras estrategias de filtrado o ranking por museo/sala/obra

## Requisitos

- Python `3.12+`
- LM Studio levantado en `http://127.0.0.1:1234`
- Un modelo de chat cargado en LM Studio
- Un modelo de embeddings cargado en LM Studio

Configuracion validada en este proyecto:

- `LM_STUDIO_CHAT_MODEL=qwen2.5-7b-instruct`
- `LM_STUDIO_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5`

## Instalacion

```bash
cd /home/eduardo/proyectos/iot/museiq/museRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuracion

El archivo `.env.example` ya esta alineado con el estado actual del proyecto:

```bash
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_CHAT_MODEL=qwen2.5-7b-instruct
LM_STUDIO_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
MUSERAG_HOST=0.0.0.0
MUSERAG_PORT=8000
MUSERAG_CHROMA_DIR=./storage/chroma
MUSERAG_COLLECTION=museiq_knowledge
MUSERAG_PDF_PATH=./tumbas-reales-sipan.pdf
MUSERAG_APP_MUSEUM_JSON=../museiqApp/content/museum.json
MUSERAG_APP_DATA_TS=../museiqApp/datos.ts
MUSERAG_TOP_K=2
MUSERAG_MAX_SOURCE_CHARS=500
MUSERAG_CHAT_MAX_TOKENS=220
CORS_ORIGINS=*
```

## Flujo de ingesta

### Paso 1: Extraer imágenes del PDF

Las imágenes del libro se extraen una sola vez y se almacenan en `assets/book_figures/` listas para consulta:

```bash
cd /home/eduardo/proyectos/iot/museiq/museRAG
source .venv/bin/activate
python extract_images.py --rebuild
```

Esto:
- Lee el PDF configurado en `.env` (`MUSERAG_PDF_PATH`)
- Extrae todas las imágenes e identifica referencias (`Fig. 01`, `Fig. 02`, etc.)
- Guarda imágenes en `assets/book_figures/` con nombres descriptivos
- Imprime un reporte de figuras extraídas

Salida esperada:
```
🖼️ Extrayendo imágenes del PDF: tumbas-reales-sipan.pdf
   ✓ Página 3: guardada Fig_01_p3.png
   ✓ Página 5: guardada Fig_02_p5.png
   ...
✅ Extracción completada: 24 imágenes guardadas
```

### Paso 2: Reconstruir la base vectorial

Ahora se indexa el contenido y se vinculan las imágenes pre-extraídas:

```bash
python ingest.py --rebuild
```

Esto:
- Lee PDF + `museum.json` + `datos.ts`
- Chunked el texto con referencias a figuras
- Vincula cada chunk a su imagen correspondiente (si existe)
- Construye los embeddings y almacena en Chroma

## Levantar la API

```bash
cd /home/eduardo/proyectos/iot/museiq/museRAG
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Interfaz web simple

Tambien puedes abrir una interfaz web minima para consultar la base vectorial sin usar `curl`:

```bash
cd /home/eduardo/proyectos/iot/museiq/museRAG
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Luego abre en tu navegador:

```bash
http://127.0.0.1:8000/
```

Desde esa interfaz puedes:

- reconstruir el indice
- hacer preguntas libres
- indicar `room_id`, `artwork_id` y `top_k`
- revisar la respuesta y los fragmentos fuente recuperados

## Endpoints

- `GET /health`
- `GET /catalog/artworks`
- `GET /media/artworks/{room}/{filename}`
- `GET /media/book_figures/{filename}` — Imágenes extraídas del PDF
- `POST /ingest/rebuild`
- `POST /chat/query`
- `POST /api/preguntar`

## Verificacion rapida

1. Levanta LM Studio con el servidor OpenAI-compatible activo.
2. Reconstruye el indice:

```bash
python ingest.py --rebuild
```

3. Inicia la API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Tambien puedes usar la interfaz web entrando a:

```bash
http://127.0.0.1:8000/
```

4. Prueba salud:

```bash
curl http://127.0.0.1:8000/health
```

5. Prueba una consulta:

```bash
curl -X POST http://127.0.0.1:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Que hay en la sala 2?","room_id":"SALA_2"}'
```

Tambien puedes usar el endpoint pensado para app movil:

```bash
curl -X POST http://127.0.0.1:8000/api/preguntar \
  -H "Content-Type: application/json" \
  -d '{
    "pregunta":"Que representa esta obra?",
    "museo":"tumbas-reales-de-sipan",
    "sala":"SALA_1",
    "obra":"obra-01"
  }'
```

6. Prueba el catalogo de imagenes:

```bash
curl http://127.0.0.1:8000/catalog/artworks
```

7. Prueba una imagen puntual:

```bash
curl http://127.0.0.1:8000/media/artworks/sala-01/01-senor-de-sipan.jpg
```

## Conectar `museiqApp`

En `museiqApp/.env`, define por ejemplo:

```bash
EXPO_PUBLIC_MUSERAG_URL=http://192.168.1.10:8000
```

Usa la IP LAN de tu PC para que el telefono pueda alcanzar el backend.

Ejemplo rapido desde React Native:

```ts
const response = await fetch(`${process.env.EXPO_PUBLIC_MUSERAG_URL}/api/preguntar`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    pregunta: "Que representa esta obra?",
    museo: "tumbas-reales-de-sipan",
    sala: "SALA_1",
    obra: "obra-01",
  }),
});

const data = await response.json();
console.log(data.respuesta);
```

## Notas sobre `museiqApp`

- La base SQLite local de la app fue versionada para regenerarse con el dataset nuevo.
- Si cambias rutas de imagen o seed curatorial y la app sigue mostrando datos viejos, reinicia Expo con cache limpia:

```bash
cd /home/eduardo/proyectos/iot/museiq/museiqApp
npx expo start -c
```

- Las imagenes de las obras se resuelven desde:
  - `museiqApp/lib/artwork-images.ts`
