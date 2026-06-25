from functools import lru_cache
from typing import List

try:  # pydantic v2 (x86): pydantic-settings is a separate package
    from pydantic_settings import BaseSettings
    _PYDANTIC_V2 = True
except ImportError:  # pydantic v1 (Jetson / Python 3.6 backport): BaseSettings is built in
    from pydantic import BaseSettings
    _PYDANTIC_V2 = False


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
    # ── Profilo prestazioni (Standard vs Optimized-TX2) ────────────────────────────
    # standard = comportamento attuale identico (buffalo_l, FP32/CUDA, full res, nessun
    # skip/tracker/batch). optimized-tx2 = ottimizzazioni Jetson, ciascun parametro OPT_*.
    performance_profile: str = "standard"     # "standard" | "optimized-tx2"
    opt_model_pack: str = "buffalo_s"         # pack InsightFace per il profilo ottimizzato
    opt_precision: str = "fp16"               # fp16 | int8 | fp32 (TensorRT)
    opt_det_width: int = 1280                 # downsample rilevamento (0,0 = risoluzione piena)
    opt_det_height: int = 720
    opt_frame_skip: int = 2                   # elabora 1 frame ogni N (ottimizzato)
    opt_tracker: bool = True                  # tracker IoU porta l'identità (salta il re-embed)
    opt_batch_embed: bool = True              # embedding di tutti i volti del frame in un passo
    trt_engine_cache_dir: str = ""            # vuoto → {data_dir}/engines (disco esterno)

    @property
    def cameras(self) -> List[str]:
        return [s.strip() for s in self.camera_sources.split(",")]

    if _PYDANTIC_V2:
        model_config = {"env_file": ".env"}
    else:  # pydantic v1 (Jetson) — stesso effetto, sintassi v1
        class Config:
            env_file = ".env"


@lru_cache(maxsize=None)
def get_settings() -> Settings:
    return Settings()
