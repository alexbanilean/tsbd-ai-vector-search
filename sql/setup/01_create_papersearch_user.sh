#!/usr/bin/env bash
# Runs once after first DB creation (official Oracle Free: /opt/oracle/scripts/setup).
set -euo pipefail
ORACLE_PWD="${ORACLE_PWD:?ORACLE_PWD must be set}"
APP_PW="${ORACLE_APP_PASSWORD:-PaperSearch_App_1}"

sqlplus -s "sys/${ORACLE_PWD}"@//localhost:1521/FREE as sysdba <<EOSQL
ALTER SESSION SET CONTAINER = FREEPDB1;
WHENEVER SQLERROR CONTINUE
CREATE USER papersearch IDENTIFIED BY "${APP_PW}"
  DEFAULT TABLESPACE USERS QUOTA UNLIMITED ON USERS;
WHENEVER SQLERROR EXIT SQL.SQLCODE
GRANT CONNECT, RESOURCE, CREATE VIEW TO papersearch;
EXIT;
EOSQL
