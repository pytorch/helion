from __future__ import annotations

from .config_fragment import BooleanFragment as BooleanFragment
from .config_fragment import EnumFragment as EnumFragment
from .config_fragment import IntegerFragment as IntegerFragment
from .config_fragment import ListOf as ListOf
from .config_fragment import PowerOfTwoFragment as PowerOfTwoFragment
from .config_spec import ConfigSpec as ConfigSpec
from .differential_evolution import (
    DifferentialEvolutionSearch as DifferentialEvolutionSearch,
)
from .effort_profile import AutotuneEffortProfile as AutotuneEffortProfile
from .effort_profile import DifferentialEvolutionConfig as DifferentialEvolutionConfig
from .effort_profile import PatternSearchConfig as PatternSearchConfig
from .effort_profile import RandomSearchConfig as RandomSearchConfig
from .finite_search import FiniteSearch as FiniteSearch
from .local_cache import LocalAutotuneCache as LocalAutotuneCache
from .local_cache import StrictLocalAutotuneCache as StrictLocalAutotuneCache
from .pattern_search import PatternSearch as PatternSearch
from .random_search import RandomSearch as RandomSearch

search_algorithms = {
    "DifferentialEvolutionSearch": DifferentialEvolutionSearch,
    "FiniteSearch": FiniteSearch,
    "PatternSearch": PatternSearch,
    "RandomSearch": RandomSearch,
}
