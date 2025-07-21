import os
from typing import Callable, List, Optional

from .collager import Collager
from .collager_config import CollagerConfig
from .audio_segment import AudioSegment
from .util import Util

def create_collage_from_files(config: CollagerConfig) -> None:
    """
    Orchestrates creating a collage from file paths.
    """
    sample_audio: AudioSegment = AudioSegment.from_file(config.sample_file)
    target_audio: AudioSegment = AudioSegment.from_file(config.target_file)

    output_audio: AudioSegment = Collager.create_collage(
        target_audio=target_audio,
        sample_audio=sample_audio,
        config=config
    )
    
    output_audio.to_file(config.outpath)

def chop_and_write_from_file(
    input_filepath: str,
    outdir: str,
    chop_length: int,
    step_ms: Optional[int] = None,
    step_factor: Optional[float] = None,
    progress_callback: Optional[Callable] = None
) -> None:
    """
    Chops a file into snippets and writes them to disk.
    """
    input_audio: AudioSegment = AudioSegment.from_file(input_filepath)

    slices: List[AudioSegment] = Util.chop_audio(
        input_audio,
        chop_length,
        step_ms=step_ms,
        step_factor=step_factor,
        progress_callback=progress_callback
    )

    for i, audio_slice in enumerate(slices):
        filename: str = f"{chop_length}ms.{i:04}.wav"
        outfile_path: str = os.path.join(outdir, filename)
        audio_slice.to_file(outfile_path)
