from __future__ import annotations

from pathlib import Path

from .schemas import ArtworkImageItem, ArtworkImageRoom, ArtworkImageCatalogResponse


def build_artwork_catalog(artworks_dir: Path) -> ArtworkImageCatalogResponse:
    rooms: list[ArtworkImageRoom] = []

    for room_dir in sorted(path for path in artworks_dir.iterdir() if path.is_dir()):
        items: list[ArtworkImageItem] = []
        for file_path in sorted(path for path in room_dir.iterdir() if path.is_file()):
            relative_path = file_path.relative_to(artworks_dir).as_posix()
            items.append(
                ArtworkImageItem(
                    filename=file_path.name,
                    room=room_dir.name,
                    relative_path=relative_path,
                    url=f"/media/artworks/{relative_path}",
                )
            )

        rooms.append(
            ArtworkImageRoom(
                room=room_dir.name,
                total=len(items),
                items=items,
            )
        )

    return ArtworkImageCatalogResponse(
        total_rooms=len(rooms),
        total_images=sum(room.total for room in rooms),
        rooms=rooms,
    )
