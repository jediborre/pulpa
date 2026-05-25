"""
Shared anti-block helpers for SofaScore API protection.

Used by both bet_monitor.py (daemon) and scraper.py (CLI/data fetching).
Prevents HTTP 403 blocks through:
  - Global 403 streak detection
  - Automatic cooldown pauses
  - Per-request spacing with jitter
  - Logging and diagnostics
"""

import threading
import time
import random

# Anti-block configuration
# AUMENTADO para Obscura (rápido) — evita detección de bot
GLOBAL_FETCH_MIN_SPACING_SECS = 1.5   # fue 2.5; reducido porque Obscura ya tiene anti-bot propio
GLOBAL_FETCH_SPACING_JITTER_SECS = 0.5  # fue 0.8
GLOBAL_403_STREAK_TRIGGER = 6
GLOBAL_403_COOLDOWN_SECS = 15 * 60  # fue 12 min; ahora 15 min para ser más conservador
GLOBAL_COOLDOWN_LOG_EVERY_SECS = 120

# Session rotation: pause briefly after N successful fetches to break activity patterns
SESSION_ROTATE_EVERY = 40        # fue 20; ampliado para no bloquear prestart con tanta frecuencia
SESSION_ROTATE_PAUSE_SECS = 45   # fue 2 min; reducido a 45s

# Global state (thread-safe)
_fetch_state_lock = threading.Lock()
_global_403_streak = 0
_global_fetch_cooldown_until = 0.0
_global_next_fetch_wall = 0.0
_last_cooldown_log_wall = 0.0
_global_session_fetch_count = 0  # counts successful fetches toward next rotation

# Optional callback for logging (set by caller, typically logging.warn)
_log_callback = None


def set_log_callback(callback):
    """Set an optional logging callback for cooldown/error events."""
    global _log_callback
    _log_callback = callback


def _log(msg):
    """Internal logging via callback if available."""
    if _log_callback:
        _log_callback(f"[anti_block] {msg}")


def is_http_403_error(msg: str) -> bool:
    """Check if error message indicates HTTP 403."""
    text = str(msg or "").lower()
    return " 403" in text or "http 403" in text


def get_global_fetch_cooldown_left() -> float:
    """Return seconds until cooldown expires (0 if not active)."""
    with _fetch_state_lock:
        return max(0.0, _global_fetch_cooldown_until - time.monotonic())


def next_global_fetch_delay_secs() -> float:
    """Return delay needed to maintain global spacing between scrapes.
    
    Call this BEFORE each API request to enforce minimum spacing with jitter.
    """
    global _global_next_fetch_wall
    with _fetch_state_lock:
        now = time.monotonic()
        due = max(0.0, _global_next_fetch_wall - now)
        slot_base = max(now, _global_next_fetch_wall)
        _global_next_fetch_wall = slot_base + GLOBAL_FETCH_MIN_SPACING_SECS
    return due + random.uniform(0.0, GLOBAL_FETCH_SPACING_JITTER_SECS)


def note_fetch_success() -> None:
    """Reset 403 streak on successful API request. Triggers session-rotation pause every N fetches."""
    global _global_403_streak, _global_session_fetch_count, _global_fetch_cooldown_until
    rotated = False
    with _fetch_state_lock:
        _global_403_streak = 0
        if SESSION_ROTATE_EVERY > 0:
            _global_session_fetch_count += 1
            if _global_session_fetch_count >= SESSION_ROTATE_EVERY:
                _global_session_fetch_count = 0
                _global_fetch_cooldown_until = max(
                    _global_fetch_cooldown_until,
                    time.monotonic() + SESSION_ROTATE_PAUSE_SECS,
                )
                rotated = True
    if rotated:
        _log(
            f"rotaci\u00f3n de sesi\u00f3n: pausa {SESSION_ROTATE_PAUSE_SECS // 60} min "
            f"tras {SESSION_ROTATE_EVERY} fetches exitosos"
        )


def note_fetch_error(msg: str, source: str = "") -> None:
    """Track 403 errors and arm cooldown when threshold hit.
    
    Parameters
    ----------
    msg : str
        Error message to check for 403 indicators.
    source : str
        Label identifying where error occurred (for logging).
    """
    global _global_403_streak, _global_fetch_cooldown_until
    now = time.monotonic()
    tripped = False
    cooldown_mins = 0.0
    streak_val = 0

    with _fetch_state_lock:
        if is_http_403_error(msg):
            _global_403_streak += 1
            streak_val = _global_403_streak
            if _global_403_streak >= GLOBAL_403_STREAK_TRIGGER:
                _global_fetch_cooldown_until = max(
                    _global_fetch_cooldown_until,
                    now + GLOBAL_403_COOLDOWN_SECS,
                )
                _global_403_streak = 0
                tripped = True
                cooldown_mins = (
                    max(0.0, _global_fetch_cooldown_until - now) / 60.0
                )
        else:
            _global_403_streak = 0

    if tripped:
        _log(
            f"pausa activada: {cooldown_mins:.1f} min tras racha HTTP 403 "
            f"(source={source}, streak={streak_val})"
        )


def maybe_log_global_cooldown(source: str = "") -> float:
    """Log cooldown status (throttled) and return seconds left.
    
    Logs are throttled to every GLOBAL_COOLDOWN_LOG_EVERY_SECS to avoid spam.
    Returns cooldown time remaining (0 if not active).
    """
    global _last_cooldown_log_wall
    now = time.monotonic()
    should_log = False
    left = 0.0

    with _fetch_state_lock:
        left = max(0.0, _global_fetch_cooldown_until - now)
        if left > 0 and (now - _last_cooldown_log_wall) >= GLOBAL_COOLDOWN_LOG_EVERY_SECS:
            _last_cooldown_log_wall = now
            should_log = True

    if should_log:
        label = f" ({source})" if source else ""
        _log(f"pausa activa{label}: ~{left / 60:.1f} min")
    return left


def wait_for_fetch_window(max_wait_secs: float = GLOBAL_403_COOLDOWN_SECS) -> bool:
    """Block until both cooldown and spacing allow a fetch.
    
    Parameters
    ----------
    max_wait_secs : float
        Maximum seconds to wait before returning False (timeout).
        Defaults to GLOBAL_403_COOLDOWN_SECS.
    
    Returns
    -------
    bool
        True if fetch window is available, False if timeout exceeded.
    """
    start = time.monotonic()
    while True:
        cooldown_left = get_global_fetch_cooldown_left()
        spacing_delay = next_global_fetch_delay_secs()
        total_wait = max(cooldown_left, spacing_delay)
        
        if total_wait <= 0:
            return True
        
        if time.monotonic() - start > max_wait_secs:
            return False
        
        # Wait a bit before checking again
        time.sleep(min(0.1, total_wait))


def reset_state() -> None:
    """Reset all anti-block state (for testing or process restarts)."""
    global _global_403_streak, _global_fetch_cooldown_until, _global_next_fetch_wall, _last_cooldown_log_wall, _global_session_fetch_count
    with _fetch_state_lock:
        _global_403_streak = 0
        _global_fetch_cooldown_until = 0.0
        _global_next_fetch_wall = 0.0
        _last_cooldown_log_wall = 0.0
        _global_session_fetch_count = 0


def get_status() -> dict:
    """Return current anti-block state for diagnostics."""
    with _fetch_state_lock:
        now = time.monotonic()
        return {
            "streak": _global_403_streak,
            "cooldown_left_secs": max(0.0, _global_fetch_cooldown_until - now),
            "next_fetch_delay_secs": max(0.0, _global_next_fetch_wall - now),
            "session_fetch_count": _global_session_fetch_count,
        }
