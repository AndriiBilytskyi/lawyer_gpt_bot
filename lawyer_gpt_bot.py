import os
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import logging
from telegram.error import NetworkError, TimedOut
from telegram.request import HTTPXRequest

# =========================
# НАСТРОЙКИ
# =========================

# Ваши переменные окружения (setx ...)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Модель: по умолчанию ставлю наиболее “универсальный” вариант.
# Если у вас есть доступ к другой модели — поменяйте через setx OPENAI_MODEL "..."
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Группа, где вы хотите использовать бота (username группы с @).
# Если оставить пустым, бот будет работать в любых чатах, где его упомянули/вызвали /ask.
raw = os.getenv("TARGET_GROUP_USERNAMES", "@advocate_ua_1,@advocate_ua_2")
TARGET_GROUP_USERNAMES = [g.strip() for g in raw.split(",") if g.strip()]

# Контакт адвоката (username без @) — кнопка для клиента
LAWYER_USERNAME = os.getenv("LAWYER_USERNAME", "Andrii_Bilytskyi").strip().lstrip("@")

# Кто может зарегистрироваться “админом” (получать лид-сводки)
ALLOWED_ADMIN_USERNAMES = {"vladmarchenko9"}  # без @

# Режим в группе:
# - "mention": отвечать только если упомянули бота /ask или ответили на сообщение бота
# - "all": отвечать на любые вопросы (обычно нужен privacy mode off или бот-админ)
GROUP_MODE = os.getenv("GROUP_MODE", "mention").strip().lower()

# Файлы хранения
DATA_DIR = os.getenv("DATA_DIR", ".").strip() or "."
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
ADMINS_FILE = os.path.join(DATA_DIR, "admins.json")
PENDING_LEADS_FILE = os.path.join(DATA_DIR, "pending_leads.json")

# Ограничение истории (чтобы не раздувать контекст и стоимость)
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # сообщений (user+assistant суммарно)

# Анти-спам
RATE_LIMIT_N = int(os.getenv("RATE_LIMIT_N", "10"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

# =========================
# INSTRUCTIONS ДЛЯ МОДЕЛИ
# =========================
INSTRUCTIONS = """
Ты — Юстин - помощник адвоката Андрея Билицкого. Задача:
1) Вежливо вести первичный диалог и дать общую ориентацию (это не юридическая консультация).
2) Собрать минимум вводных для адвоката без чувствительных персональных данных.
3) Когда нужен адвокат — направить клиента к адвокату и подготовить сводку (handoff).

Правила:
- Всегда указывай, что ты Юстин-помощник, а не адвокат; ответы — общая информация.
- Не проси паспортные данные, адрес, номера документов, даты рождения. Если нужно — проси обезличенно.
- Пиши на языке пользователя (RU/UA/DE/En).
- Если вопрос из группы — предложи продолжить в личке.
- Если вопрос требует анализа документов/сроков/суда/стратегии или срочно — делай handoff.

В конце КАЖДОГО ответа добавь мета-блок строго так:
[[META]]{"handoff": false, "summary_for_lawyer": "..."}[[/META]]
summary_for_lawyer: 1–3 предложения по сути, без персональных данных.
"""

# =========================
# REGEX / ВНУТРЕННЕЕ
# =========================
META_RE = re.compile(r"\[\[META\]\](.*?)\[\[/META\]\]", re.DOTALL)

QUESTION_RE = re.compile(
    r"\?|"
    r"\b(как|что|почему|зачем|когда|где|куда|сколько|можно ли|"
    r"подскажите|підкажіть|посоветуйте|порадьте|порекомендуйте|рекомендуйте|"
    r"нужен|ищу|кто может|кто знает|есть ли)\b",
    re.IGNORECASE
)

def is_question(text: str) -> bool:
    if not text:
        return False
    return bool(QUESTION_RE.search(text.strip()))

# Локальные триггеры handoff (даже если модель не поставит handoff=true)
HANDOFF_HINT_RE = re.compile(
    r"\b(консультац|стоим|цена|гонорар|адвокат|представител|договор|суд|срок|уголовн|развод|опека|внж|aufenthalt)\b",
    re.IGNORECASE,
)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


SESSIONS: Dict[str, Any] = load_json(SESSIONS_FILE, {})
ADMINS: Dict[str, Any] = load_json(ADMINS_FILE, {"admin_ids": []})


def now_ts() -> float:
    return time.time()


def is_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return bool(QUESTION_RE.search(t)) or ("?" in t)


def extract_meta(text: str) -> Tuple[str, Dict[str, Any]]:
    meta = {"handoff": False, "summary_for_lawyer": ""}
    m = META_RE.search(text or "")
    if not m:
        return (text or "").strip(), meta

    raw = m.group(1).strip()
    reply = (text[:m.start()] + text[m.end():]).strip()

    try:
        meta = json.loads(raw)
        meta.setdefault("handoff", False)
        meta.setdefault("summary_for_lawyer", "")
    except Exception:
        meta = {"handoff": False, "summary_for_lawyer": ""}

    return reply, meta


def build_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Связаться с адвокатом", url=f"https://t.me/{LAWYER_USERNAME}")]]
    )


def get_user_state(user_id: int) -> Dict[str, Any]:
    sid = str(user_id)
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "history": [],     # [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
            "rate_ts": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    return SESSIONS[sid]


def rate_limited(state: Dict[str, Any]) -> bool:
    ts = state.get("rate_ts", [])
    now = now_ts()
    ts = [t for t in ts if now - t < RATE_LIMIT_WINDOW_SEC]
    ts.append(now)
    state["rate_ts"] = ts
    return len(ts) > RATE_LIMIT_N


def in_target_group(update) -> bool:
    """
    Возвращает True, если сообщение пришло из одной из целевых групп.
    Поддерживает TARGET_GROUP_USERNAMES (список) и падение назад на TARGET_GROUP_USERNAME (строка).
    """

    chat = getattr(update.effective_chat, "username", None) or ""
    chat_at = f"@{chat}".lower().strip() if chat else ""

    # 1) Новый вариант: список групп
    if "TARGET_GROUP_USERNAMES" in globals() and TARGET_GROUP_USERNAMES:
        targets = {g.lower().strip() for g in TARGET_GROUP_USERNAMES if g and g.strip()}
        return chat_at in targets

    # 2) Старый вариант: одна группа (если вдруг где-то ещё используется)
    if "TARGET_GROUP_USERNAME" in globals() and TARGET_GROUP_USERNAME:
        return chat_at == TARGET_GROUP_USERNAME.lower().strip()

    # Если ничего не задано — считаем, что это не целевая группа
    return False


def group_should_respond(update: Update, bot_username: str) -> bool:
    msg = update.message
    if not msg or not msg.text:
        return False

    text = msg.text.strip()
    text_lc = text.lower()
    bot_at = f"@{bot_username.lower()}" if bot_username else ""

    # 1) Явный вызов всегда разрешаем
    if text_lc.startswith("/ask"):
        return True
    if bot_at and bot_at in text_lc:
        return True

    # 2) Ответ на сообщение бота — разрешаем
    if (
        msg.reply_to_message
        and msg.reply_to_message.from_user
        and msg.reply_to_message.from_user.is_bot
    ):
        return True

    # 3) Если это reply на ЧУЖОЕ сообщение — НЕ отвечаем (это межпользовательский диалог)
    if msg.reply_to_message is not None:
        return False

    # 4) Режим "all": отвечаем только на "самостоятельные" вопросы (не reply) и только если это вопрос
    if GROUP_MODE == "all":
        return is_question(text)

    # 5) Режим mention: всё остальное игнорируем
    return False


def strip_call_prefix(text: str, bot_username: str) -> str:
    t = text.strip()
    if t.lower().startswith("/ask"):
        return t[4:].strip()
    # убираем упоминание бота, если есть
    if bot_username:
        t = re.sub(rf"@{re.escape(bot_username)}\b", "", t, flags=re.IGNORECASE).strip()
    return t


def build_message_link(chat, message_id: int) -> str:
    # Для публичных групп:
    if getattr(chat, "username", None):
        return f"https://t.me/{chat.username}/{message_id}"
    # Для приватных супергрупп часто работает /c/ (если chat_id начинается с -100)
    cid = str(chat.id)
    if cid.startswith("-100"):
        return f"https://t.me/c/{cid[4:]}/{message_id}"
    return "(ссылка недоступна)"


async def send_long(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    # Telegram ограничивает длину сообщения; на практике безопасно резать около 3500-3800
    chunk_size = 3800
    for i in range(0, len(text), chunk_size):
        await context.bot.send_message(chat_id=chat_id, text=text[i:i + chunk_size])


async def send_to_admins(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    admin_ids: List[int] = ADMINS.get("admin_ids", [])
    if not admin_ids:
        save_json(PENDING_LEADS_FILE, {"saved_at": datetime.now().isoformat(timespec="seconds"), "lead": text})
        return
    for aid in admin_ids:
        try:
            await send_long(context, aid, text)
        except Exception:
            pass


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте. Я Юстин - помощник адвоката Андрея Билицкого.\n"
        "Я могу дать первичную ориентацию (общая информация, не юридическая консультация) и при необходимости "
        "перевести вас к адвокату.\n\n"
        "Пожалуйста, не отправляйте паспортные данные, адрес и номера документов.",
        reply_markup=build_keyboard(),
    )


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await update.message.reply_text(f"Ваш numeric user_id: {u.id}")


async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    username = (u.username or "").lower()
    allowed = {x.lower() for x in ALLOWED_ADMIN_USERNAMES}

    if username in allowed:
        admin_ids: List[int] = ADMINS.get("admin_ids", [])
        if u.id not in admin_ids:
            admin_ids.append(u.id)
            ADMINS["admin_ids"] = admin_ids
            save_json(ADMINS_FILE, ADMINS)
        await update.message.reply_text("Готово. Вы зарегистрированы как получатель лид-сводок.")
    else:
        await update.message.reply_text("Доступ к /admin ограничен.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    SESSIONS.pop(str(u.id), None)
    save_json(SESSIONS_FILE, SESSIONS)
    await update.message.reply_text("Диалог сброшен. Напишите вопрос заново.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
        await update.message.reply_text(
            "Бот не настроен: проверьте переменные окружения TELEGRAM_BOT_TOKEN и OPENAI_API_KEY."
        )
        return

    if not in_target_group(update):
        return

    chat = update.effective_chat
    user = update.effective_user
    bot_username = (context.bot.username or "").lstrip("@")

    text = update.message.text.strip()

    # В группах отвечаем только по правилам
    if chat.type in ("group", "supergroup"):
        if not group_should_respond(update, bot_username):
            return
        text = strip_call_prefix(text, bot_username)

    state = get_user_state(user.id)
    if rate_limited(state):
        await update.message.reply_text("Слишком много сообщений за минуту. Попробуйте чуть позже.")
        save_json(SESSIONS_FILE, SESSIONS)
        return

    history: List[Dict[str, str]] = state.get("history", [])
    history = history[-MAX_TURNS:] if MAX_TURNS > 0 else history

    local_handoff = bool(HANDOFF_HINT_RE.search(text))

    try:
        prompt_context = history + [{"role": "user", "content": text}]

        resp = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=prompt_context,
            store=False,
        )

        raw = (resp.output_text or "").strip()
        reply_text, meta = extract_meta(raw)

        # Обновляем историю
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply_text})
        state["history"] = history
        save_json(SESSIONS_FILE, SESSIONS)

        # Если вопрос из группы — мягко в личку
        if chat.type in ("group", "supergroup"):
            deep_link = f"https://t.me/{bot_username}?start=from_group" if bot_username else ""
            reply_text += "\n\nЧтобы не обсуждать детали публично, лучше продолжить в личке с ботом."
            if deep_link:
                reply_text += f"\nПерейти в личку: {deep_link}"

        await update.message.reply_text(reply_text, reply_markup=build_keyboard())

        handoff = bool(meta.get("handoff")) or local_handoff
        if handoff:
            summary = (meta.get("summary_for_lawyer") or "").strip()
            if not summary:
                summary = text[:700]

            msg_link = build_message_link(chat, update.message.message_id)

            lead = (
                "Лид от AI-бота\n"
                f"Пользователь: {user.full_name} (@{user.username}) id={user.id}\n"
                f"Чат: {chat.type} {chat.title or ''}\n"
                f"Ссылка на сообщение: {msg_link}\n"
                f"Сводка: {summary}\n"
                f"Исходный текст: {text}\n"
                f"Контакт адвоката: https://t.me/{LAWYER_USERNAME}"
            )
            await send_to_admins(context, lead)

            await update.message.reply_text(
                f"Для детальной консультации лучше перейти к адвокату: https://t.me/{LAWYER_USERNAME}",
                reply_markup=build_keyboard(),
            )

    except Exception as e:
        await update.message.reply_text(f"Техническая ошибка: {e}")


import traceback
from telegram.error import NetworkError, TimedOut

logger = logging.getLogger(__name__)

# Для редкого "мягкого рестарта" при серии сетевых ошибок:
_LAST_NET_RESTART_TS = 0.0

async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _LAST_NET_RESTART_TS

    err = context.error

    # 1) Сетевые ошибки polling — ожидаемые на VPN/нестабильной сети.
    # Не печатаем traceback, чтобы не “пугать” и не засорять консоль.
    if isinstance(err, (NetworkError, TimedOut)):
        logger.warning("Telegram network issue (polling): %s", err)

        # 2) Опционально: если сеть “сыпется” — перезапустить polling не чаще 1 раза в 60 сек.
        # Это помогает быстрее восстановиться после смены VPN.
        now = time.time()
        if now - _LAST_NET_RESTART_TS > 60:
            _LAST_NET_RESTART_TS = now
            try:
                # run_polling() завершится, ваш цикл main() поднимет его заново
                await context.application.stop()
            except Exception:
                pass
        return

    # 3) Все остальные ошибки — логируем с traceback.
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    logger.error("Unhandled exception while handling an update:\n%s", tb)


def build_app() -> Application:
    # Таймауты снижают вероятность "подвисаний" на нестабильной сети/VPN
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
    )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Важно: регистрируем error handler
    app.add_error_handler(global_error_handler)

    return app


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Нет TELEGRAM_BOT_TOKEN в переменных окружения.")
    if not OPENAI_API_KEY:
        raise SystemExit("Нет OPENAI_API_KEY в переменных окружения.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ВАЖНО: не печатать токен в URL-логах httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    backoff = 5  # стартовая пауза при сетевых сбоях
    while True:
        app = build_app()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot started. Model={OPENAI_MODEL}, GROUP_MODE={GROUP_MODE}")

        try:
            app.run_polling(
                drop_pending_updates=True,
                allowed_updates=None,
            )
            backoff = 5  # если штатно остановился — сброс backoff

        except KeyboardInterrupt:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stopped by user.")
            break

        except (NetworkError, TimedOut) as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polling network error: {e}. Restart in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)  # максимум 2 минуты

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fatal error: {type(e).__name__}: {e}. Restart in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
