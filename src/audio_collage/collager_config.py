from dataclasses import dataclass, field
from strenum import StrEnum
from typing import List, Optional

@dataclass(frozen=True)
class CollagerConfig:
    """A single object to hold all collage generation parameters."""

    DeclickFn = StrEnum('Declickfn', {k: k for k in ['sigmoid', 'linear']})
    DistanceFn = StrEnum('DistanceFn', {k: k for k in ['mfcc', 'fast_mfcc', 'mean_mfcc', 'mfcc_cosine']})

    # File paths
    target_file: str = None
    sample_file: str = None
    outpath: str = None

    # Collage parameters
    windows: List[int] = field(default_factory=lambda: [800, 400, 200, 100, 50])
    distance_fn: DistanceFn = DistanceFn.mfcc

    # Declicking parameters
    declick_fn: Optional[DeclickFn] = DeclickFn.sigmoid
    declick_ms: int = 0

    # Chopping parameters
    step_ms: Optional[int] = None
    step_factor: Optional[float] = None

    # Progress callback
    progress_callback: callable = None

    def __post_init__(self):
        if self.step_ms is not None and self.step_factor is not None:
            raise ValueError("Cannot specify both 'step_ms' and 'step_factor'.")
