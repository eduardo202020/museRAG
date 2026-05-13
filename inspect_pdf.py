from __future__ import annotations

import argparse
from pathlib import Path
from pypdf import PdfReader

def inspect_pdf(pdf_path: Path) -> None:
    """Inspect PDF structure and image objects."""
    reader = PdfReader(str(pdf_path))
    
    print(f"📄 Inspeccionando: {pdf_path.name}")
    print(f"📊 Total de páginas: {len(reader.pages)}\n")
    
    total_xobjects = 0
    total_images = 0
    image_types = {}

    for page_index, page in enumerate(reader.pages, start=1):
        if "/Resources" not in page:
            continue
            
        if "/XObject" not in page["/Resources"]:
            continue

        xobject = page["/Resources"]["/XObject"].get_object()
        page_images = 0
        
        for obj_name, obj in xobject.items():
            total_xobjects += 1
            
            if obj.get("/Subtype") != "/Image":
                continue
            
            page_images += 1
            total_images += 1
            
            # Extract metadata
            width = obj.get("/Width", "?")
            height = obj.get("/Height", "?")
            color_space = obj.get("/ColorSpace", "?")
            filter_type = obj.get("/Filter", "None")
            bits = obj.get("/BitsPerComponent", "?")
            
            img_type = f"{filter_type}_{width}x{height}"
            image_types[img_type] = image_types.get(img_type, 0) + 1
            
            print(f"  Página {page_index}, imagen {page_images}:")
            print(f"    • Tamaño: {width} x {height}")
            print(f"    • ColorSpace: {color_space}")
            print(f"    • Filter: {filter_type}")
            print(f"    • BitsPerComponent: {bits}")

    print(f"\n📈 Estadísticas:")
    print(f"  • Total XObjects: {total_xobjects}")
    print(f"  • Total imágenes: {total_images}")
    print(f"\n🖼️  Tipos de imágenes:")
    for img_type, count in sorted(image_types.items()):
        print(f"  • {img_type}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspecciona la estructura de imágenes en un PDF")
    parser.add_argument("pdf", nargs="?", help="Ruta al PDF (usa MUSERAG_PDF_PATH del .env si no se especifica)")
    args = parser.parse_args()
    
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        from app.config import get_settings
        settings = get_settings()
        pdf_path = settings.pdf_path
    
    if not pdf_path.exists():
        print(f"❌ Error: PDF no encontrado: {pdf_path}")
        exit(1)
    
    inspect_pdf(pdf_path)
