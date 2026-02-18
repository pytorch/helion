from __future__ import annotations

from typing import TYPE_CHECKING

from .aot_cache import AOTAutotuneCache as AOTAutotuneCache
from .config_fragment import BooleanFragment as BooleanFragment
from .config_fragment import EnumFragment as EnumFragment
from .config_fragment import IntegerFragment as IntegerFragment
from .config_fragment import ListOf as ListOf
from .config_fragment import PowerOfTwoFragment as PowerOfTwoFragment
from .config_spec import ConfigSpec as ConfigSpec
from .de_surrogate_hybrid import DESurrogateHybrid as DESurrogateHybrid
from .differential_evolution import (
    DifferentialEvolutionSearch as DifferentialEvolutionSearch,
)
from .effort_profile import AutotuneEffortProfile as AutotuneEffortProfile
from .effort_profile import DifferentialEvolutionConfig as DifferentialEvolutionConfig
from .effort_profile import PatternSearchConfig as PatternSearchConfig
from .effort_profile import RandomSearchConfig as RandomSearchConfig
from .external import UserConfigSpec as UserConfigSpec
from .external import autotune as autotune
from .finite_search import FiniteSearch as FiniteSearch
from .local_cache import LocalAutotuneCache as LocalAutotuneCache
from .local_cache import StrictLocalAutotuneCache as StrictLocalAutotuneCache
from .modal_search import ModalSearch as ModalSearch
from .pattern_search import InitialPopulationStrategy as InitialPopulationStrategy
from .pattern_search import PatternSearch as PatternSearch
from .random_search import RandomSearch as RandomSearch
from .surrogate_pattern_search import LFBOPatternSearch

if TYPE_CHECKING:
    from .base_search import BaseSearch

search_algorithms: dict[str, type[BaseSearch]] = {
    "DESurrogateHybrid": DESurrogateHybrid,
    "LFBOPatternSearch": LFBOPatternSearch,
    "DifferentialEvolutionSearch": DifferentialEvolutionSearch,
    "FiniteSearch": FiniteSearch,
    "ModalSearch": ModalSearch,
    "PatternSearch": PatternSearch,
    "RandomSearch": RandomSearch,
}

cache_classes = {
    "LocalAutotuneCache": LocalAutotuneCache,
    "StrictLocalAutotuneCache": StrictLocalAutotuneCache,
    "AOTAutotuneCache": AOTAutotuneCache,
}
