from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

@dataclass
class CollageProgressState:
    """
    A dataclass to hold the state of a collage generation process.
    """
    class Task(Enum):
        CHOPPING = auto()
        INDEXING = auto()
        SELECTING = auto()
        CONCATENATING = auto()

    task: Task
    total_steps: Optional[int] = None
    current_step: Optional[int] = None
    message: Optional[str] = None
    starting: bool = False
    advance: int = 0
    completed: bool = False

    def __post_init__(self) -> None:
        if self.task not in CollageProgressState.Task.__members__.values():
            raise ValueError(f"Invalid task: {self.task}")
