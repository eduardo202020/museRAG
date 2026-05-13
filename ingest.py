from __future__ import annotations

import argparse

from app.config import get_settings
from app.rag_service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye el indice de MuseRAG")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruye la coleccion completa")
    args = parser.parse_args()

    if not args.rebuild:
        parser.error("Usa --rebuild para reconstruir la base vectorial.")

    settings = get_settings()
    service = RagService(settings)
    indexed = service.rebuild_index()
    print(f"Indexacion completada. Documentos cargados: {indexed}")


if __name__ == "__main__":
    main()
