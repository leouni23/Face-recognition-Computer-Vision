from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/face_id"
    biometric_secret_key: str = "change_me_generate_a_random_hex_key"
    camera_sources: str = "0"
    match_threshold: float = 0.5
    use_gpu: bool = True
    data_retention_days: int = 365
    position_log_interval: float = 1.0  # seconds between saved position points per person/camera
    log_level: str = "INFO"

    @property
    def cameras(self) -> List[str]:
        return [s.strip() for s in self.camera_sources.split(",")]

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
