"""Authentication for the admin UI.

Two modes:
  - AUTH_ENABLED=true  (default)   API key required via X-API-Key header, Bearer
                                   token, or session cookie set via /login.
  - AUTH_ENABLED=false             Open access. For isolated on-prem networks.

The key is read from ADMIN_API_KEY. If auth is on and no key is set, a random
one is generated and printed once at startup.
"""
import logging
import os
import secrets
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

COOKIE_NAME = "vllm_admin_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


AUTH_ENABLED: bool = _env_bool("AUTH_ENABLED", True)
ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "").strip()
_GENERATED_KEY: bool = False

if AUTH_ENABLED and not ADMIN_API_KEY:
    ADMIN_API_KEY = secrets.token_urlsafe(32)
    _GENERATED_KEY = True


def log_startup_banner() -> None:
    if not AUTH_ENABLED:
        logger.warning("AUTH DISABLED (AUTH_ENABLED=false). Admin API is open — ensure the network is trusted.")
        return
    if _GENERATED_KEY:
        logger.warning("No ADMIN_API_KEY set; generated one for this session.")
        logger.warning("    Admin API key: %s", ADMIN_API_KEY)
        logger.warning("    Set ADMIN_API_KEY to keep it stable across restarts.")
    else:
        logger.info("Auth enabled. Admin API key loaded from ADMIN_API_KEY env var.")


def _extract_key(request: Request) -> Optional[str]:
    header = request.headers.get("X-API-Key")
    if header:
        return header.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return cookie.strip()
    return None


def is_authenticated(request: Request) -> bool:
    if not AUTH_ENABLED:
        return True
    supplied = _extract_key(request)
    if not supplied:
        return False
    return secrets.compare_digest(supplied, ADMIN_API_KEY)


def verify_key(candidate: str) -> bool:
    if not AUTH_ENABLED:
        return True
    if not candidate:
        return False
    return secrets.compare_digest(candidate.strip(), ADMIN_API_KEY)


# Paths that must work without auth regardless of mode.
_PUBLIC_PATHS = {
    "/login",
    "/api/auth/login",
    "/api/auth/status",
    "/api/auth/logout",
    "/healthz",
}


def is_public_path(path: str) -> bool:
    return path in _PUBLIC_PATHS


async def auth_middleware(request: Request, call_next):
    """Gate every request that isn't a public auth path."""
    if not AUTH_ENABLED:
        return await call_next(request)

    path = request.url.path
    if is_public_path(path):
        return await call_next(request)

    if is_authenticated(request):
        return await call_next(request)

    # Browser navigations (any non-API GET: "/", "/cluster", "/static/*", …) get
    # redirected to the login page — carrying ?next= so login returns them here;
    # API callers get a 401.
    if request.method == "GET" and not path.startswith("/api/"):
        from urllib.parse import quote
        target = path + (("?" + request.url.query) if request.url.query else "")
        return RedirectResponse(url="/login?next=" + quote(target, safe=""),
                                status_code=302)
    return JSONResponse(
        status_code=401,
        content={"error": "Authentication required"},
    )
