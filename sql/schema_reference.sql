-- Reference DDL for Oracle AI Vector Search (23ai+).
-- Application applies live schema via `python -m papersearch.cli init`.
-- Includes `category` for arXiv-style hybrid filters.

/*
CREATE TABLE papers (
  paper_id         VARCHAR2(64) PRIMARY KEY,
  title            VARCHAR2(4000) NOT NULL,
  abstract_text    CLOB NOT NULL,
  authors            VARCHAR2(2000),
  year_published     NUMBER(4),
  venue              VARCHAR2(500),
  doi                VARCHAR2(256),
  category           VARCHAR2(500),
  combined_text      CLOB NOT NULL,
  embedding          VECTOR(384, FLOAT32) NOT NULL,
  created_at         TIMESTAMP(9) DEFAULT SYSTIMESTAMP NOT NULL
);
*/
