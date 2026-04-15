from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import oracledb

from papersearch.config import Settings, get_settings

logger = logging.getLogger(__name__)

_pool: oracledb.ConnectionPool | None = None

# Pool: min=0 avoids blocking pool creation when DB is still starting; longer waits for cold Oracle.
_POOL_TCP_TIMEOUT = 20.0
_POOL_WAIT_S = 120


def init_pool(settings: Settings | None = None) -> oracledb.ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    cfg = settings or get_settings()
    _pool = oracledb.create_pool(
        user=cfg.oracle_user,
        password=cfg.oracle_password,
        dsn=cfg.oracle_dsn,
        min=0,
        max=8,
        increment=1,
        getmode=oracledb.POOL_GETMODE_TIMEDWAIT,
        wait_timeout=_POOL_WAIT_S,
        timeout=120,
        tcp_connect_timeout=_POOL_TCP_TIMEOUT,
    )
    logger.info("Oracle connection pool ready (user=%s dsn=%s)", cfg.oracle_user, cfg.oracle_dsn)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close(force=True)
        _pool = None


def format_connect_help(settings: Settings) -> str:
    u, d = settings.oracle_user, settings.oracle_dsn
    return (
        f"Could not open an Oracle session (user={u!r}, dsn={d!r}).\n"
        "Checklist:\n"
        "  • Container up: docker compose ps\n"
        "  • Wait for DB: python3 scripts/wait_for_oracle.py\n"
        "  • Passwords: ORACLE_APP_PASSWORD / ORACLE_PWD in compose match PAPERSEARCH_ORACLE_*\n"
        "  • Service name: PAPERSEARCH_ORACLE_DSN=localhost:1521/FREEPDB1\n"
    )


@contextmanager
def direct_connection(settings: Settings | None = None) -> Iterator[oracledb.Connection]:
    """
    Single dedicated session (no pool). Prefer this for CLI so failures surface as ORA-xxxxx
    instead of DPY-4005 pool wait timeouts when the listener is down or the PDB is not open yet.
    """
    cfg = settings or get_settings()
    conn = oracledb.connect(
        user=cfg.oracle_user,
        password=cfg.oracle_password,
        dsn=cfg.oracle_dsn,
        tcp_connect_timeout=_POOL_TCP_TIMEOUT,
    )
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def connection(settings: Settings | None = None) -> Iterator[oracledb.Connection]:
    pool = init_pool(settings)
    conn = pool.acquire()
    try:
        yield conn
    finally:
        pool.release(conn)
