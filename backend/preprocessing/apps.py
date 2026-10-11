from django.apps import AppConfig


class PreprocessingConfig(AppConfig):
    name = "preprocessing"
    verbose_name = "Layer 1: Preprocessing"

    def ready(self) -> None:
        # Fail at startup, not on the first upload, if settings.PREPROCESSING
        # contains an unknown or invalid option.
        from .services import get_default_config

        get_default_config()
