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
        config=config
    )
    
    output_audio.to_file(config.outpath)
