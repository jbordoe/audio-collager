from dataclasses import dataclass
from enum import Enum, auto

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
    total_steps: int = None
    current_step: int = None
    message: str = None
    starting: bool = False
    completed: bool = False

    def __post_init__(self):
        if self.task not in CollageProgressState.Task.__members__.values():
            raise ValueError(f"Invalid task: {self.task}")
