from .collager import Collager
from .audio_segment import AudioSegment

def create_collage_from_files(
    target_file: str,
    sample_file: str,
    outpath: str,
    declick_fn: Collager.DeclickFn,
    declick_ms: int,
    distance_fn: Collager.DistanceFn,
    step_ms: int,
    step_factor: float
):
    """
    Orchestrates creating a collage from file paths.
    """
    sample_audio = AudioSegment.from_file(sample_file)
    target_audio = AudioSegment.from_file(target_file)

    output_audio = Collager.create_collage(
        target_audio=target_audio,
        sample_audio=sample_audio,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn,
        step_ms=step_ms,
        step_factor=step_factor
    )
    
    output_audio.to_file(outpath)
