from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TileInterface:
    """Base interface for tile objects in Helion."""

    @property
    def index(self) -> torch.Tensor:
        """
        Alias for :func:`~helion.language.tile_index`, which retrieves a tensor containing the offsets for a tile.
        """
        from .tile_ops import tile_index

        return tile_index(self)

    @property
    def begin(self) -> int:
        """
        Alias for :func:`~helion.language.tile_begin`, which retrieves the start offset of a tile.
        """
        from .tile_ops import tile_begin

        return tile_begin(self)

    @property
    def end(self) -> int:
        """
        Alias for :func:`~helion.language.tile_end`, which retrieves the end offset of a tile.
        """
        from .tile_ops import tile_end

        return tile_end(self)

    @property
    def block_size(self) -> int:
        """
        Alias for :func:`~helion.language.tile_block_size`, which retrieves the block_size of a tile.
        """
        from .tile_ops import tile_block_size

        return tile_block_size(self)

    @property
    def count(self) -> int:
        """
        Alias for :func:`~helion.language.tile_count`, which retrieves the number of tiles.
        """
        from .tile_ops import tile_count

        return tile_count(self)

    @property
    def id(self) -> int:
        """
        Alias for :func:`~helion.language.tile_id`, which retrieves the id of a tile.
        """
        from .tile_ops import tile_id

        return tile_id(self)
