"""Notifications for unknown-subject detections + enrollment by message.

`Notifier` is the swappable messaging interface (a WhatsApp backend could be added
without touching the detection code). `TelegramNotifier` (pyTelegramBotAPI) is the
concrete backend; `NullNotifier` is the no-op default when disabled.

Privacy: alerts are TEXT ONLY — a reference id (camera + time + short ref), never an
image, consistent with the system's no-image design. The unknown's mean embedding is
held in a pending buffer keyed by that ref; on "Autorizza" + a name the subject is
enrolled (encrypted template + consent) and the live pipeline reloads immediately.
"""
import secrets
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from loguru import logger

from database.repository import PersonRepository
from database.session import get_session

_NAME_OK = lambda s: bool(s) and len(s) <= 100 and all(c.isalnum() or c in " '.-" for c in s)


class Notifier(ABC):
    enabled = False

    def start(self) -> None:
        """Begin background work (e.g. polling). No-op for most backends."""

    @abstractmethod
    def alert_unknown(self, camera_id: str, mean_embedding: np.ndarray) -> None:
        ...


class NullNotifier(Notifier):
    enabled = False

    def alert_unknown(self, camera_id: str, mean_embedding: np.ndarray) -> None:
        return


def _enroll_mean(mean_embedding: np.ndarray, name: str) -> str:
    """Enroll (or update) `name` from a mean embedding; returns an action word."""
    norm = float(np.linalg.norm(mean_embedding))
    emb = (mean_embedding / norm).astype(np.float32) if norm else mean_embedding.astype(np.float32)
    with get_session() as session:
        repo = PersonRepository(session)
        person = repo.get_by_name(name)
        action = "aggiornata" if person else "iscritta"
        if not person:
            person = repo.create(name)
        repo.give_consent(person)
        repo.add_template(person, emb)
    # reload templates so the new person is recognised immediately
    try:
        from web.broadcaster import broadcaster
        if broadcaster.pipeline is not None:
            broadcaster.pipeline.force_reload()
    except Exception:
        pass
    return action


class TelegramNotifier(Notifier):
    enabled = True
    _PENDING_TTL = 180.0  # secondi: un alert non risolto blocca i nuovi, poi scade

    def __init__(self, token: str, chat_id: str):
        import telebot

        self._telebot = telebot
        self.bot = telebot.TeleBot(token, parse_mode=None)
        self.chat_id = str(chat_id)
        self._lock = threading.Lock()
        self._pending: Dict[str, dict] = {}        # ref → {mean, camera, ts}
        self._awaiting_name: Dict[str, str] = {}    # chat_id → ref
        self._register()

    def start(self) -> None:
        threading.Thread(
            target=self.bot.infinity_polling,
            kwargs={"timeout": 20, "long_polling_timeout": 20},
            daemon=True, name="telegram-bot",
        ).start()
        logger.info("TelegramNotifier: polling avviato")

    def alert_unknown(self, camera_id: str, mean_embedding: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            # drop stale pendings, then send only if nothing is awaiting your action
            for r in [r for r, p in self._pending.items() if now - p["ts"] > self._PENDING_TTL]:
                self._pending.pop(r, None)
                self._awaiting_name = {k: v for k, v in self._awaiting_name.items() if v != r}
            if self._pending:
                return  # un alert è già in attesa di Autorizza/Nega/nome → non inondare
            ref = secrets.token_hex(3)
            self._pending[ref] = {"mean": mean_embedding, "camera": camera_id, "ts": now}
        markup = self._telebot.types.InlineKeyboardMarkup()
        markup.add(
            self._telebot.types.InlineKeyboardButton("✅ Autorizza", callback_data=f"ok:{ref}"),
            self._telebot.types.InlineKeyboardButton("🚫 Nega", callback_data=f"no:{ref}"),
        )
        text = (
            "👤 Soggetto sconosciuto rilevato\n"
            f"Camera: {camera_id}\n"
            f"Ora: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Rif: {ref}\n\n"
            "Autorizzare l'iscrizione?"
        )
        try:
            self.bot.send_message(self.chat_id, text, reply_markup=markup)
        except Exception as exc:
            logger.warning(f"TelegramNotifier: invio alert fallito ({exc})")
            with self._lock:
                self._pending.pop(ref, None)  # non lasciare un pending bloccante

    # ── handlers ──────────────────────────────────────────────────────────────
    def _register(self):
        @self.bot.callback_query_handler(func=lambda c: True)
        def _on_callback(call):
            if str(call.message.chat.id) != self.chat_id:
                return
            action, _, ref = call.data.partition(":")
            with self._lock:
                pending = self._pending.get(ref)
            if pending is None:
                self.bot.answer_callback_query(call.id, "Riferimento scaduto.")
                return
            if action == "no":
                with self._lock:
                    self._pending.pop(ref, None)
                self.bot.answer_callback_query(call.id, "Ignorato.")
                self.bot.send_message(self.chat_id, f"🚫 Rif {ref} ignorato.")
            elif action == "ok":
                with self._lock:
                    self._awaiting_name[str(call.message.chat.id)] = ref
                self.bot.answer_callback_query(call.id, "Invia il nome.")
                self.bot.send_message(self.chat_id, f"✍️ Invia il NOME per il soggetto (rif {ref}):")

        @self.bot.message_handler(func=lambda m: True, content_types=["text"])
        def _on_text(msg):
            chat = str(msg.chat.id)
            if chat != self.chat_id:
                return
            with self._lock:
                ref = self._awaiting_name.pop(chat, None)
                pending = self._pending.pop(ref, None) if ref else None
            if not ref or pending is None:
                return  # not awaiting a name → ignore
            name = (msg.text or "").strip()
            if not _NAME_OK(name):
                self.bot.send_message(self.chat_id, "Nome non valido. Riprova: ✍️ invia di nuovo il nome.")
                with self._lock:  # keep waiting
                    self._awaiting_name[chat] = ref
                    self._pending[ref] = pending
                return
            # Prefer a FRESH embedding from the current frame (single face) over the older
            # buffered mean → the template matches the live appearance, so the camera switches
            # from "Sconosciuto" to the name almost immediately.
            emb = pending["mean"]
            try:
                from core.detector import detect_and_encode
                from web.broadcaster import broadcaster
                frame = broadcaster.get_raw_frame(pending.get("camera"))
                if frame is not None:
                    faces = detect_and_encode(frame)
                    if len(faces) == 1:
                        emb = faces[0][1]
            except Exception:
                pass
            try:
                action = _enroll_mean(emb, name)
                self.bot.send_message(self.chat_id, f"✅ '{name}' {action}. Riconoscimento attivo subito.")
                logger.success(f"[Telegram] '{name}' {action} (rif {ref})")
            except Exception as exc:
                logger.error(f"[Telegram] enrollment fallito: {exc}")
                self.bot.send_message(self.chat_id, f"⚠ Errore durante l'iscrizione: {exc}")


# ── module-level swappable singleton ────────────────────────────────────────────
_notifier: Notifier = NullNotifier()


def get_notifier() -> Notifier:
    return _notifier


def set_notifier(n: Notifier) -> None:
    global _notifier
    _notifier = n


def build_notifier(settings) -> Notifier:
    """Construct the notifier from settings; NullNotifier if disabled or misconfigured."""
    if not getattr(settings, "telegram_enabled", False):
        return NullNotifier()
    token = getattr(settings, "telegram_bot_token", "")
    chat_id = getattr(settings, "telegram_chat_id", "")
    if not token or not chat_id:
        logger.warning("Telegram abilitato ma TELEGRAM_BOT_TOKEN/CHAT_ID mancanti — disattivato")
        return NullNotifier()
    try:
        return TelegramNotifier(token, chat_id)
    except Exception as exc:
        logger.error(f"TelegramNotifier non inizializzabile ({exc}) — disattivato")
        return NullNotifier()
