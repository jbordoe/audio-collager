from dataclasses import dataclass, field
from typing import List, Optional

from .collager import Collager

@dataclass(frozen=True)
class CollageConfig:
    """A single object to hold all collage generation parameters."""

    # File paths
    target_file: str
    sample_file: str
    outpath: str

    # Collage parameters
    windows: List[int] = field(default_factory=lambda: [800, 400, 200, 100, 50])
    distance_fn: Collager.DistanceFn = Collager.DistanceFn.mfcc

    # Declicking parameters
    declick_fn: Optional[Collager.DeclickFn] = Collager.DeclickFn.sigmoid
    declick_ms: int = 20

    # Chopping parameters
    step_ms: Optional[int] = None
    step_factor: Optional[float] = 0.5 # Default to 50% overlap

    def __post_init__(self):
        if self.step_ms is not None and self.step_factor is not None:
            raise ValueError("Cannot specify both 'step_ms' and 'step_factor'.")

