"""Centralized utilities for tile operations."""

from typing import Literal, Optional
import torch
from .compile_environment import CompileEnvironment


class TileOperations:
    """Centralized utilities for tile-related operations."""
    
    @staticmethod
    def is_tile_index_tensor(fake_tensor: torch.Tensor, env: Optional[CompileEnvironment] = None) -> bool:
        """Check if tensor represents a tile.index result."""
        if env is None:
            env = CompileEnvironment.current()
        return (fake_tensor.ndim == 1 and 
                fake_tensor.dtype == env.settings.index_dtype and
                len(fake_tensor.shape) == 1 and
                hasattr(fake_tensor.shape[0], '_sympy_'))
    
    @staticmethod
    def find_tile_block_id(tile_var: torch.SymInt, env: Optional[CompileEnvironment] = None) -> Optional[int]:
        """Find block_id for a given tile variable."""
        if env is None:
            env = CompileEnvironment.current()
        
        tile_sympy = tile_var._sympy_() if hasattr(tile_var, '_sympy_') else tile_var
        for block_id, block_info in enumerate(env.block_sizes):
            block_sympy = block_info.var._sympy_() if hasattr(block_info.var, '_sympy_') else block_info.var
            if tile_sympy == block_sympy:
                return block_id
        return None
    
    @staticmethod
    def find_tile_block_id_for_tensor(fake_tensor: torch.Tensor, env: Optional[CompileEnvironment] = None) -> Optional[int]:
        """Find block_id for a tile index tensor by matching its shape."""
        if not TileOperations.is_tile_index_tensor(fake_tensor, env):
            return None
            
        if env is None:
            env = CompileEnvironment.current()
            
        shape_var = fake_tensor.shape[0]
        if hasattr(shape_var, '_sympy_'):
            for block_id, block_info in enumerate(env.block_sizes):
                if hasattr(block_info.var, '_sympy_') and shape_var._sympy_() == block_info.var._sympy_():
                    return block_id
        return None
    
    @staticmethod
    def register_tile_tensor(tensor: torch.Tensor, block_id: int, tile_var: torch.SymInt) -> None:
        """Register a tile.index tensor with its origin."""
        from .host_function import HostFunction
        from .variable_origin import TileOrigin
        
        host_fn = HostFunction.current()
        if host_fn is not None:
            origin = TileOrigin(kind='index', block_id=block_id, tile=tile_var)
            host_fn.tensor_to_origin[tensor] = origin
    
    @staticmethod
    def get_tile_property_block_id(symint: torch.SymInt, property_name: Literal['tile_begin', 'tile_end', 'tile_id']) -> Optional[int]:
        """Get block_id for tile properties like tile_begin, tile_end, tile_id."""
        from .host_function import HostFunction
        from .variable_origin import UnbackedSymIntOrigin
        
        expr = symint._sympy_()
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        
        if origin_info and isinstance(origin_info.origin, UnbackedSymIntOrigin):
            cache_key = origin_info.origin.cache_key
            if len(cache_key) >= 2 and cache_key[0] == property_name:
                # Find the block_id for this tile
                tile_var = cache_key[1]
                return TileOperations.find_tile_block_id(tile_var)
        return None