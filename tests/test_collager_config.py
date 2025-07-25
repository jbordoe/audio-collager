from audio_collage.collager_config import CollagerConfig
import pytest

def test_collager_config():
    config = CollagerConfig(
        target_file='test/data/target.wav',
        sample_file='test/data/sample.wav',
        outpath='test/data/output.wav',
        windows=[800, 400, 200, 100, 50],
        distance_fn=CollagerConfig.DistanceFn.mfcc,
        declick_fn=CollagerConfig.DeclickFn.sigmoid,
        declick_ms=20,
        step_ms=None,
        step_factor=0.5
    )

    assert config.target_file == 'test/data/target.wav'
    assert config.sample_file == 'test/data/sample.wav'
    assert config.outpath == 'test/data/output.wav'
    assert config.windows == [800, 400, 200, 100, 50]
    assert config.distance_fn == CollagerConfig.DistanceFn.mfcc
    assert config.declick_fn == CollagerConfig.DeclickFn.sigmoid
    assert config.declick_ms == 20
    assert config.step_ms is None
    assert config.step_factor == 0.5

def test_collage_config_errors():
    # Cannot specify both step_ms and step_factor
    with pytest.raises(ValueError):
        CollagerConfig(
            target_file='test/data/target.wav',
            sample_file='test/data/sample.wav',
            outpath='test/data/output.wav',
            step_ms=100,
            step_factor=0.5
        )
