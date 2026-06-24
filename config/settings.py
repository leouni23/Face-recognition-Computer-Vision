from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/face_id"
    biometric_secret_key: str = "change_me_generate_a_random_hex_key"
    camera_sources: str = "0"
    match_threshold: float = 0.5
    det_threshold: float = 0.5   # confidenza minima del rilevatore SCRFD (default InsightFace)
    min_face_px: int = 80        # scarta volti più bassi di N px (riflessi/specchio, volti lontani)
    use_gpu: bool = True
    web_password: str = ""  # Basic Auth per la Web UI; vuota = nessuna autenticazione (solo localhost!)
    data_retention_days: int = 365
    position_log_interval: float = 1.0  # seconds between saved position points per person/camera
    metrics_enabled: bool = True  # pipeline timing instrumentation (overhead trascurabile)
    data_dir: str = "data"  # base dir per artefatti persistenti (DB, validation, benchmark, log)
    validation_dir: str = ""  # override root artefatti validazione (vuoto → data_dir/validation); può puntare a un disco esterno. Il DB resta su storage interno.
    validation_record_video: bool = False  # default validazione senza video (nessuna immagine su disco); per-sessione si può riattivare
    # Notifiche Telegram (alert soggetti sconosciuti + enrollment da messaggio)
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    unknown_alert_cooldown: float = 10.0      # secondi minimi tra un alert e il successivo per camera
    unknown_alert_min_duration: float = 1.0   # secondi di "sconosciuto" CONTINUO prima di allertare
    unknown_alert_warmup: float = 15.0        # nessun alert nei primi N secondi (warm-up camera)
    unknown_alert_min_samples: int = 3        # (legacy) non più usato per il gate
    log_level: str = "INFO"

    @property
    def cameras(self) -> List[str]:
        return [s.strip() for s in self.camera_sources.split(",")]

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
