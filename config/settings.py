import os
from functools import lru_cache
from pathlib import Path
from typing import List

try:  # pydantic v2 (x86): pydantic-settings is a separate package
    from pydantic_settings import BaseSettings
    _PYDANTIC_V2 = True
except ImportError:  # pydantic v1 (Jetson / Python 3.6 backport): BaseSettings is built in
    from pydantic import BaseSettings
    _PYDANTIC_V2 = False


def _load_runtime_env() -> None:
    """Load runtime overrides persisted by the UI (profile switch, threshold) from
    ${DATA_DIR}/runtime.env INTO os.environ, so Settings() picks them up as normal env vars.

    Critical: the boot must NEVER depend on a file written at runtime. This is best-effort — if
    python-dotenv is missing OR the file is absent/unreadable, we skip silently and the app boots
    from the compose environment. (Previously _persist_env wrote /app/.env which pydantic re-read
    at boot, hard-requiring python-dotenv → ModuleNotFoundError crash-loop after any UI switch.)
    """
    path = Path(os.environ.get("DATA_DIR", "data")) / "runtime.env"
    try:
        if not path.is_file():
            return
        from dotenv import dotenv_values  # optional dep; absence must not break boot
        for key, value in (dotenv_values(str(path)) or {}).items():
            if value is not None:
                os.environ[key] = value  # persisted user choice overrides the compose default
    except ImportError:
        pass  # python-dotenv not installed → ignore runtime overrides, boot from env
    except OSError:
        pass


_load_runtime_env()


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
    opt_reembed_every: int = 10               # re-embedda un volto tracciato ogni N frame (0=mai) → la distanza a video si aggiorna
    opt_batch_embed: bool = True              # embedding di tutti i volti del frame in un passo
    trt_engine_cache_dir: str = ""            # vuoto → {data_dir}/engines (disco esterno)
    # Fallback memoria: se onnxruntime r32.7 non libera le arene native al cambio profilo, esci
    # in modo pulito e lascia che Docker (restart: unless-stopped) riparta sul nuovo profilo.
    profile_switch_restart: bool = False      # PROFILE_SWITCH_RESTART=true per abilitare

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
