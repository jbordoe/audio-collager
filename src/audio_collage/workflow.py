from .collager import Collager
from .collager_config import CollagerConfig
from .audio_segment import AudioSegment

def create_collage_from_files(config: CollagerConfig):
    """
    Orchestrates creating a collage from file paths.
    """
    sample_audio = AudioSegment.from_file(config.sample_file)
    target_audio = AudioSegment.from_file(config.target_file)

    output_audio = Collager.create_collage(
        target_audio=target_audio,
        sample_audio=sample_audio,
        declick_fn=config.declick_fn,
        declick_ms=config.declick_ms,
        distance_fn=config.distance_fn,
        step_ms=config.step_ms,
        step_factor=config.step_factor
    )
    
    output_audio.to_file(config.outpath)
