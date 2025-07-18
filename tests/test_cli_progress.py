from audio_collage.collage_progress_state import CollageProgressState
from audio_collage.cli_progress import CLIProgress
from unittest.mock import MagicMock, patch

@patch('rich.progress.Progress')
def test_init(mock_progress):
    """
    Test that the constructor sets the correct attributes.
    """
    mock_progress = MagicMock()
    progress = CLIProgress()

    assert mock_progress.start.called_once()
    assert progress.task_ids == {}

@patch('rich.progress.Progress')
def test_update(mock_progress):
    """
    Test that the update method calls the correct methods.
    """
    mock_progress = MagicMock()
    progress = CLIProgress()

    for task in CollageProgressState.Task.__members__.values():
        state = CollageProgressState(
            task,
            starting=True,
            current_step=0,
            total_steps=100,
            message="Hello world!"
        )
        progress.update(state)

        assert mock_progress.start.called_once()
        assert mock_progress.add_task.called_once_with(
            description=str,
            total=100,
            completed=0
        )
        assert mock_progress.update.called_once_with(
            progress.task_ids[state.task],
            completed=0
        )

def test_update_with_completed_task():
    """
    Test that the update method calls the correct methods when the task is completed.
    """
    mock_task_id = 123
    mock_progress = MagicMock()
    mock_progress.add_task.return_value = mock_task_id
    progress = CLIProgress()

    progress.update(CollageProgressState(
        CollageProgressState.Task.CHOPPING,
        completed=False,
        current_step=55,
    ))
    state = CollageProgressState(
        CollageProgressState.Task.CHOPPING,
        completed=True,
        current_step=100,
    )

    progress.update(state)

    assert mock_progress.update.called_once_with(
        mock_task_id,
        completed=100
    )

def test_update_with_nonexistent_task():
    """
    Test that the update method raises an exception when the task is invalid.
    """
    mock_task_id = 123
    mock_progress = MagicMock()
    mock_progress.add_task.return_value = mock_task_id
    progress = CLIProgress()

    state = CollageProgressState(
        CollageProgressState.Task.CHOPPING,
        completed=True,
        current_step=100,
    )

    progress.update(state)

    assert mock_progress.update.not_called()
