from rich.progress import Progress
from audio_collage.collage_progress_state import CollageProgressState

class CLIProgress:
    """
    Handles callbacks for the CLI progress bar.
    """
    def __init__(self):
        self.progress = Progress()
        self.progress.start()
        self.task_ids = {}

    def update(self, state: CollageProgressState):
        if state.starting:
            self._start_task(state)
        elif state.completed:
            self._complete_task(state)
        else:
            self._update_task(state)

    def _update_task(self, state: CollageProgressState):
        if state.task not in self.task_ids:
            self._start_task(state)

        task_id = self.task_ids[state.task]
        self.progress.update(task_id, completed=state.current_step)

    def _start_task(self, state: CollageProgressState):
        if state.task == CollageProgressState.Task.CHOPPING:
            description = "Chopping sample audio ..."
        elif state.task == CollageProgressState.Task.INDEXING:
            description = "Indexing sample audio ..."
        elif state.task == CollageProgressState.Task.SELECTING:
            description = "Selecting samples ..."
        elif state.task == CollageProgressState.Task.CONCATENATING:
            description = "Concatenating samples ..."

        task_id = self.progress.add_task(
            description=description,
            total=state.total_steps,
            completed=state.current_step
        )
        self.task_ids[state.task] = task_id

    def _complete_task(self, state: CollageProgressState):
        if state.task not in self.task_ids:
            return
        task_id = self.task_ids[state.task]
        self.progress.update(task_id, completed=state.current_step)
        self.task_ids[state.task] = None

    def __del__(self):
        self.progress.stop()
