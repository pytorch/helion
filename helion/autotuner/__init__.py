from __future__ import annotations

from .config_fragment import (
    BooleanFragment as BooleanFragment,
    EnumFragment as EnumFragment,
    IntegerFragment as IntegerFragment,
    ListOf as ListOf,
    PowerOfTwoFragment as PowerOfTwoFragment,
)
from .config_spec import ConfigSpec as ConfigSpec
from .de_surrogate_hybrid import DESurrogateHybrid as DESurrogateHybrid
from .differential_evolution import (
    DifferentialEvolutionSearch as DifferentialEvolutionSearch,
)
from .effort_profile import (
    AutotuneEffortProfile as AutotuneEffortProfile,
    DifferentialEvolutionConfig as DifferentialEvolutionConfig,
    PatternSearchConfig as PatternSearchConfig,
    RandomSearchConfig as RandomSearchConfig,
)
from .finite_search import FiniteSearch as FiniteSearch
from .local_cache import (
    LocalAutotuneCache as LocalAutotuneCache,
    StrictLocalAutotuneCache as StrictLocalAutotuneCache,
)
from .pattern_search import PatternSearch as PatternSearch
from .random_search import RandomSearch as RandomSearch
from .ucb_pattern_search import UCBPatternSearch

search_algorithms = {
    "DESurrogateHybrid": DESurrogateHybrid,
    "UCBPatternSearch": UCBPatternSearch,
    "DifferentialEvolutionSearch": DifferentialEvolutionSearch,
    "FiniteSearch": FiniteSearch,
    "PatternSearch": PatternSearch,
    "RandomSearch": RandomSearch,
}

cache_classes = {
    "LocalAutotuneCache": LocalAutotuneCache,
    "StrictLocalAutotuneCache": StrictLocalAutotuneCache,
}
