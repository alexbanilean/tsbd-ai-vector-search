"""One-off SYSDBA helper: raise VECTOR_MEMORY_SIZE in the PDB (fixes ORA-51962 for HNSW)."""

from __future__ import annotations

import os
import re
import sys
from typing import Final

import oracledb

_VECTOR_SIZE_RE: Final[re.Pattern[str]] = re.compile(r"^\d+[MG]$", re.IGNORECASE)
_ORA_CODE_RE: Final[re.Pattern[str]] = re.compile(r"ORA[- ](\d+)", re.IGNORECASE)

_DEFAULT_PDB = "FREEPDB1"

_VECTOR_MEMORY_CAP_HELP: Final[str] = (
    "Oracle cannot increase VECTOR_MEMORY_SIZE for this PDB (ORA-51955 / similar). "
    "This is common on Oracle Database Free with tight RAM. You can still run the demo: "
    "continue with init / seed (or reindex-vector). Search uses exact top-k over "
    "VECTOR_DISTANCE when no HNSW index exists. Optionally set "
    "PAPERSEARCH_USE_APPROXIMATE_FETCH=false in .env so settings match that mode."
)


class VectorMemoryNotConfigurableError(Exception):
    """PDB cannot raise VECTOR_MEMORY_SIZE (e.g. ORA-51955 on capped Oracle Free)."""


def _ora_codes(exc: BaseException) -> set[int]:
    codes: set[int] = set()
    for arg in getattr(exc, "args", ()) or ():
        c = getattr(arg, "code", None)
        if isinstance(c, int):
            codes.add(c)
    codes.update(int(m.group(1)) for m in _ORA_CODE_RE.finditer(str(exc)))
    return codes


def _is_vector_memory_cap_error(exc: BaseException) -> bool:
    codes = _ora_codes(exc)
    if 51955 in codes:
        return True
    # ORA-02097 often accompanies ORA-51955 when the vector pool cannot grow.
    if 2097 in codes and "VECTOR" in str(exc).upper():
        return True
    return False


def parse_vector_memory_size(size: str) -> str:
    size_u = size.strip().upper()
    if not _VECTOR_SIZE_RE.fullmatch(size_u):
        raise ValueError("size must look like 512M or 1G")
    return size_u


def set_vector_memory_size(
    *,
    size: str,
    sys_password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    cdb_service: str | None = None,
    pdb_name: str | None = None,
    tcp_timeout: float = 20.0,
) -> None:
    """
    Connect as SYS to the CDB service, switch to PDB, run ALTER SYSTEM for vector pool.

    Raises oracledb.Error on failure.
    """
    pw = (
        sys_password
        or os.environ.get("ORACLE_SYS_PASSWORD")
        or os.environ.get("ORACLE_PWD")
        or os.environ.get("ORACLE_PASSWORD")
        or "PaperSearch_Sys_1"
    )
    h = host or os.environ.get("ORACLE_HOST", "localhost")
    p = int(port or os.environ.get("ORACLE_PORT", "1521"))
    cdb = (cdb_service or os.environ.get("ORACLE_CDB_SERVICE", "FREE")).strip()
    pdb = (pdb_name or os.environ.get("ORACLE_PDB", _DEFAULT_PDB)).strip().upper()
    if pdb != _DEFAULT_PDB and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,29}$", pdb):
        raise ValueError(f"Refusing unsafe PDB name: {pdb!r}")

    size_u = parse_vector_memory_size(size)

    dsn = f"{h}:{p}/{cdb}"
    conn = oracledb.connect(
        user="sys",
        password=pw,
        dsn=dsn,
        mode=oracledb.AUTH_MODE_SYSDBA,
        tcp_connect_timeout=tcp_timeout,
    )
    last_err: oracledb.DatabaseError | None = None
    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER SESSION SET CONTAINER = {pdb}")
            for scope in ("BOTH", "MEMORY"):
                try:
                    cur.execute(
                        f"ALTER SYSTEM SET vector_memory_size = {size_u} SCOPE={scope}"
                    )
                    return
                except oracledb.DatabaseError as e:
                    last_err = e
        if last_err is not None:
            if _is_vector_memory_cap_error(last_err):
                raise VectorMemoryNotConfigurableError(_VECTOR_MEMORY_CAP_HELP) from last_err
            raise last_err
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Set Oracle VECTOR_MEMORY_SIZE (SYSDBA, PDB)")
    p.add_argument(
        "--size",
        default=os.environ.get("ORACLE_VECTOR_MEMORY_SIZE", "512M"),
        help="Pool size, e.g. 512M or 1G (default: env ORACLE_VECTOR_MEMORY_SIZE or 512M)",
    )
    args = p.parse_args(argv)
    try:
        set_vector_memory_size(size=args.size)
    except VectorMemoryNotConfigurableError as e:
        print(e, file=sys.stderr)
        return 3
    except oracledb.Error as e:
        codes = _ora_codes(e) if isinstance(e, oracledb.DatabaseError) else set()
        authish = codes & {1017, 28000, 28009, 12170}
        if authish or 1031 in codes:
            print(
                "Failed (credentials / SYSDBA). Set ORACLE_SYS_PASSWORD (compose maps it "
                "to the DB SYS password), or ORACLE_PWD / ORACLE_PASSWORD if you set them "
                "explicitly.",
                file=sys.stderr,
            )
        else:
            print("Failed to set VECTOR_MEMORY_SIZE.", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    pdb = os.environ.get("ORACLE_PDB", _DEFAULT_PDB).strip().upper()
    print(f"VECTOR_MEMORY_SIZE set to {parse_vector_memory_size(args.size)} in PDB {pdb}.")
    print("Re-run: python3 -m papersearch.cli reindex-vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
