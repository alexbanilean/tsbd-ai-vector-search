#!/usr/bin/env python3
"""Poll Oracle Free until python-oracledb can open a session (Makefile demo target)."""
from __future__ import annotations

import os
import sys
import time

import oracledb

user = os.environ.get("PAPERSEARCH_ORACLE_USER", "papersearch")
password = os.environ.get("PAPERSEARCH_ORACLE_PASSWORD", "PaperSearch_App_1")
dsn = os.environ.get("PAPERSEARCH_ORACLE_DSN", "localhost:1521/FREEPDB1")
deadline = time.time() + float(os.environ.get("ORACLE_WAIT_SECONDS", "900"))


def main() -> int:
    print(f"Waiting for Oracle at {dsn} (user={user}) …")
    while time.time() < deadline:
        try:
            conn = oracledb.connect(user=user, password=password, dsn=dsn)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM dual")
                assert cur.fetchone()[0] == 1
            conn.close()
            print("Oracle is accepting connections.")
            return 0
        except oracledb.Error:
            time.sleep(3)
    print("Timed out waiting for Oracle.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
