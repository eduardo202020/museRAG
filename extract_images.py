from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.loaders import extract_pdf_images_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae imágenes del libro PDF para usarlas en las respuestas de MuseRAG"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruye la carpeta de imágenes (elimina las existentes y reextrae)"
    )
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = settings.pdf_path
    figures_dir = Path(settings.base_dir) / "assets" / "book_figures"

    if args.rebuild:
        print(f"[INFO] Limpiando directorio: {figures_dir}")
        if figures_dir.exists():
            for img in figures_dir.glob("*.png"):
                img.unlink()
                print(f"   Eliminada: {img.name}")

    print(f"\n[INFO] Procesando PDF: {pdf_path}")
    print(f"[INFO] Destino: {figures_dir}\n")

    image_map = extract_pdf_images_batch(pdf_path, figures_dir)

    print("[INFO] Resumen:")
    print(f"   Total de imágenes: {len(image_map)}")
    print(f"   Directorio: {figures_dir}")
    
    if image_map:
        print("\n[INFO] Figuras extraidas:")
        for fig_num in sorted(image_map.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            print(f"   - Fig. {fig_num}: {image_map[fig_num]}")

    print("\n[OK] Las imagenes estan listas para consulta mediante RAG.")
    print(f"Ejecuta: python ingest.py --rebuild")


if __name__ == "__main__":
    main()
