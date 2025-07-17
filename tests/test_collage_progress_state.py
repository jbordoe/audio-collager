from audio_collage.collage_progress_state import CollageProgressState
import pytest

def test_init():
    """
    Test that the constructor sets the correct attributes.
    """
    task = CollageProgressState.Task.CHOPPING
    total_steps = 100
    current_step = 50
    message = "Hello world!"
    state = CollageProgressState(task, total_steps, current_step, message)

    assert state.task == task
    assert state.total_steps == total_steps
    assert state.current_step == current_step
    assert state.message == message

def test_init_with_invalid_task():
    """
    Test that the constructor raises an exception when the task is invalid.
    """
    with pytest.raises(ValueError):
        CollageProgressState(11, 100, 50, "Hello world!")
