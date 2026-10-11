import pytest

from preprocessing.pipeline.config import ConfigError, PreprocessingConfig


def test_defaults_follow_the_paper():
    config = PreprocessingConfig()
    assert config.denoiser == "drunet"  # best performer in the paper
    assert config.render_dpi == 300  # dataset resolution
    assert config.max_sigma == 0.20  # highest validated noise level
    assert config.deskew_enabled and config.crop_to_content


def test_round_trips_through_dict():
    config = PreprocessingConfig(denoiser="tv", fixed_sigma=0.1)
    assert PreprocessingConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "overrides",
    [
        {"denoiser": "bm3d"},
        {"sigma_mode": "guess"},
        {"render_dpi": 10},
        {"fixed_sigma": 0},
        {"min_sigma": 0.3, "max_sigma": 0.2},
        {"tile_size": 500},  # not a multiple of 8
        {"tile_size": 64, "tile_overlap": 32},
        {"tv_lambda": -1.0},
        {"max_pages": 0},
    ],
)
def test_rejects_invalid_values(overrides):
    with pytest.raises(ConfigError):
        PreprocessingConfig(**overrides)


def test_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        PreprocessingConfig.from_dict({"denoizer": "tv"})
    with pytest.raises(ConfigError):
        PreprocessingConfig().with_overrides({"nope": 1})


def test_with_overrides_revalidates():
    with pytest.raises(ConfigError):
        PreprocessingConfig().with_overrides({"denoiser": "bm3d"})
    assert PreprocessingConfig().with_overrides({"denoiser": "tv"}).denoiser == "tv"
