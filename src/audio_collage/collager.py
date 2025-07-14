from .util import Util
from .audio_mapper import AudioMapper
from .audio_file import AudioFile

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from strenum import StrEnum

class Collager:
    DeclickFn = StrEnum('Declickfn', {k: k for k in ['sigmoid', 'linear']})
    DistanceFn = StrEnum('DistanceFn', {k: k for k in ['mfcc', 'fast_mfcc', 'mean_mfcc']})

    @staticmethod
    def create_collage(
        target_file: str,
        sample_file: str,
        outpath: str,
        declick_fn: DeclickFn,
        declick_ms: int,
        distance_fn: DistanceFn
    ):
        """
        This is the core logic for creating a collage.
        """
        default_dc_ms = {
            'sigmoid': 20,
            'linear': 70,
        }
        if declick_fn:
            declick_ms = declick_ms or default_dc_ms[declick_fn]
        else:
            declick_ms = 0

        sample_audio = Util.read_audio(sample_file)
        target_audio = Util.read_audio(target_file)

        windows = [500, 200, 100, 50]
        windows = [i + declick_ms for i in windows]

        dist_fn_map: Dict[str, Callable[[AudioFile, AudioFile], float]] = {
            'mfcc': Util.mfcc_dist,
            'fast_mfcc': Util.fast_mfcc_dist,
            'mean_mfcc': Util.mean_mfcc_dist,
        }

        selected_distance_fn = dist_fn_map.get(distance_fn.value)
        if not selected_distance_fn:
            print(f'[yellow]Invalid distance function [yellow bold]{distance_fn}[yellow]!')
            raise typer.Exit(code=1)

        mapper = AudioMapper(sample_audio, target_audio, distance_fn=selected_distance_fn)
        selected_snippets = mapper.map_audio(
            windows=windows,
            overlap_ms=declick_ms,
        )

        output_audio = Util.concatenate_audio(
            track(selected_snippets, description="[cyan]Concatenating samples..."),
            declick_fn=declick_fn,
            declick_ms=declick_ms
        )

        print(f'[cyan]Saving collage to [yellow]{outpath}[cyan]...')
        Util.save_audio(output_audio, outpath)

        print('[green bold]Done!')
