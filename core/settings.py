"""Server and channel settings (``/config``), stored in SQLite.

A setting is either per channel (with a server-wide default an admin can also
set) or per server. Lookups go channel → parent channel (for threads) →
server default → built-in default from the environment, and are served from
memory: the whole table is loaded at startup and every write updates both.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from config import (
    CHAT_NAME_TRIGGER_ENABLED,
    CHAT_REPLY_STYLE,
    CHAT_THREAD_CHANNELS,
)

logger = logging.getLogger(__name__)

GUILD_SCOPE = 0  # channel_id used for server-level rows
_MAX_TEXT = 500


@dataclass(frozen=True)
class Setting:
    key: str
    scope: str  # 'channel' (server default allowed) or 'guild'
    kind: str  # 'bool' | 'choice' | 'text' | 'channel' | 'names'
    default: Any
    label: str
    help: str
    choices: tuple[str, ...] = ()


SETTINGS: dict[str, Setting] = {s.key: s for s in (
    Setting(
        'respond', 'channel', 'choice', 'on', 'Quando responder',
        'on: menções, respostas, chamadas pelo nome e threads do bot; mentions: só menções '
        'e respostas às mensagens do bot; off: não conversa neste canal (comandos de barra continuam).',
        ('on', 'mentions', 'off'),
    ),
    Setting(
        'name_trigger', 'channel', 'bool', CHAT_NAME_TRIGGER_ENABLED, 'Responder ao nome sem @',
        'Atende quando chamado pelo nome ("quill, o que acha?") mesmo sem menção.',
    ),
    Setting(
        'threads', 'channel', 'choice', 'auto', 'Threads',
        'auto: conversas longas ou fora do assunto vão para uma thread; always: toda conversa '
        'abre uma thread; never: sempre responde no canal.',
        ('auto', 'always', 'never'),
    ),
    Setting(
        'reply_style', 'channel', 'choice', CHAT_REPLY_STYLE if CHAT_REPLY_STYLE in ('auto', 'plain', 'embed') else 'auto',
        'Formato das respostas',
        'auto: mensagem normal, com embed só para resumos, relatórios e respostas longas; '
        'plain: sempre mensagem normal; embed: sempre embed.',
        ('auto', 'plain', 'embed'),
    ),
    Setting(
        'log_detection', 'channel', 'bool', True, 'Detectar logs e Spark',
        'Reage a links de log (mclo.gs, pastebin), arquivos .log/.txt e relatórios Spark.',
    ),
    Setting(
        'instructions', 'channel', 'text', '', 'Instruções do canal',
        'Orientações para o bot neste canal (ex.: "seja direto e técnico", "aqui é papo livre").',
    ),
    Setting(
        'aliases', 'guild', 'names', (), 'Apelidos do bot',
        'Outros nomes pelos quais o bot atende, separados por vírgula.',
    ),
    Setting(
        'wrapped', 'guild', 'choice', 'off', 'Wrapped automático',
        'Publica uma retrospectiva do servidor toda semana (weekly) ou todo mês (monthly).',
        ('off', 'weekly', 'monthly'),
    ),
    Setting(
        'wrapped_channel', 'guild', 'channel', '', 'Canal do Wrapped',
        'Canal onde a retrospectiva automática é publicada.',
    ),
    Setting(
        'voice', 'guild', 'bool', False, 'Conversa por voz',
        'Deixa o bot entrar nas calls com /voz entrar e conversar por voz. Enquanto ele está '
        'na call, o áudio de quem fala vai para a IA de voz (Qwen, da Alibaba Cloud).',
    ),
)}


class SettingError(ValueError):
    """User-facing validation error."""


def parse_value(setting: Setting, raw: Any) -> str:
    """Validate *raw* for *setting* and return its stored text form."""
    text = str(raw if raw is not None else '').strip()
    if setting.kind == 'bool':
        low = text.lower()
        if low in ('1', 'true', 'yes', 'sim', 'on', 'ligado', 'ativado'):
            return 'true'
        if low in ('0', 'false', 'no', 'nao', 'não', 'off', 'desligado', 'desativado'):
            return 'false'
        raise SettingError(f'{setting.label}: use sim/não.')
    if setting.kind == 'choice':
        low = text.lower()
        if low not in setting.choices:
            raise SettingError(f'{setting.label}: use um de {", ".join(setting.choices)}.')
        return low
    if setting.kind == 'channel':
        digits = ''.join(c for c in text if c.isdigit())
        if text and not digits:
            raise SettingError(f'{setting.label}: informe um canal.')
        return digits
    if setting.kind == 'names':
        names = [n.strip() for n in text.split(',') if n.strip()]
        if any(len(n) < 3 or len(n) > 32 for n in names):
            raise SettingError(f'{setting.label}: cada nome deve ter de 3 a 32 caracteres.')
        return ', '.join(dict.fromkeys(names))
    if len(text) > _MAX_TEXT:
        raise SettingError(f'{setting.label}: no máximo {_MAX_TEXT} caracteres.')
    return text


def coerce(setting: Setting, stored: str | None) -> Any:
    if stored is None:
        return list(setting.default) if setting.kind == 'names' else setting.default
    if setting.kind == 'bool':
        return stored == 'true'
    if setting.kind == 'names':
        return [n.strip() for n in stored.split(',') if n.strip()]
    return stored


def display_value(setting: Setting, value: Any) -> str:
    if setting.kind == 'bool':
        return 'sim' if value else 'não'
    if setting.kind == 'channel':
        return f'<#{value}>' if value else '—'
    if setting.kind == 'names':
        return ', '.join(value) if value else '—'
    return str(value) if value not in (None, '') else '—'


class SettingsStore:
    """In-memory settings backed by SQLite (``SETTINGS_DB_PATH``)."""

    def __init__(self, path: str):
        self.path = path
        self._values: dict[tuple[int, int, str], str] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def load(self) -> None:
        """Create the table if needed and load every row into memory."""
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        con = self._connect()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, channel_id, key)
                )""")
            con.commit()
            rows = con.execute('SELECT guild_id, channel_id, key, value FROM settings').fetchall()
        finally:
            con.close()
        with self._lock:
            self._values = {(int(g), int(c), k): v for g, c, k, v in rows}
            self._loaded = True

    # --- Raw access ---

    def raw(self, guild_id: int, channel_id: int, key: str) -> str | None:
        return self._values.get((int(guild_id), int(channel_id), key))

    def write(self, guild_id: int, channel_id: int, key: str, value: str | None, *, updated_by: str = '') -> None:
        """Store (or with ``None`` remove) one value; synchronous, call via a thread."""
        k = (int(guild_id), int(channel_id), key)
        con = self._connect()
        try:
            if value is None:
                con.execute(
                    'DELETE FROM settings WHERE guild_id=? AND channel_id=? AND key=?', k,
                )
            else:
                con.execute(
                    'INSERT OR REPLACE INTO settings (guild_id, channel_id, key, value, updated_by, updated_at) '
                    'VALUES (?,?,?,?,?,?)',
                    (*k, value, updated_by, datetime.datetime.now(datetime.timezone.utc).isoformat()),
                )
            con.commit()
        finally:
            con.close()
        with self._lock:
            if value is None:
                self._values.pop(k, None)
            else:
                self._values[k] = value

    # --- Resolved lookups ---

    @staticmethod
    def _channel_chain(channel: Any) -> list[int]:
        ids: list[int] = []
        cid = getattr(channel, 'id', None)
        if cid is not None:
            ids.append(int(cid))
        parent = getattr(channel, 'parent_id', None)
        if parent:
            ids.append(int(parent))
        return ids

    def get(self, guild_id: int | None, channel: Any, key: str) -> Any:
        """Effective value of *key* in *channel* (a channel, thread or ``None``)."""
        setting = SETTINGS[key]
        if guild_id is None:
            return coerce(setting, None)
        if setting.scope == 'channel':
            for cid in self._channel_chain(channel):
                stored = self.raw(guild_id, cid, key)
                if stored is not None:
                    return coerce(setting, stored)
        stored = self.raw(guild_id, GUILD_SCOPE, key)
        if stored is not None:
            return coerce(setting, stored)
        if key == 'threads' and any(cid in CHAT_THREAD_CHANNELS for cid in self._channel_chain(channel)):
            return 'always'
        return coerce(setting, None)

    def for_channel(self, guild: Any, channel: Any) -> dict[str, Any]:
        """Every channel-scoped setting resolved for *channel*."""
        gid = getattr(guild, 'id', None)
        return {
            key: self.get(gid, channel, key)
            for key, s in SETTINGS.items() if s.scope == 'channel'
        }

    def source(self, guild_id: int, channel: Any, key: str) -> str:
        """Where the effective value comes from: 'canal', 'canal pai', 'servidor' or 'padrão'."""
        chain = self._channel_chain(channel) if SETTINGS[key].scope == 'channel' else []
        for i, cid in enumerate(chain):
            if self.raw(guild_id, cid, key) is not None:
                return 'canal' if i == 0 else 'canal pai'
        if self.raw(guild_id, GUILD_SCOPE, key) is not None:
            return 'servidor'
        return 'padrão'

    def guilds_with(self, key: str) -> dict[int, str]:
        """guild_id → stored server-level value of *key*."""
        return {g: v for (g, c, k), v in self._values.items() if k == key and c == GUILD_SCOPE}

    # --- Internal state (not user settings), e.g. when the last Wrapped went out ---

    def state(self, guild_id: int, key: str) -> str | None:
        return self.raw(guild_id, GUILD_SCOPE, f'_{key}')

    def set_state(self, guild_id: int, key: str, value: str) -> None:
        self.write(guild_id, GUILD_SCOPE, f'_{key}', value, updated_by='bot')
