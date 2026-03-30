-- ══════════════════════════════════════════════════════
--  NeoRx — PostgreSQL Schema
-- ══════════════════════════════════════════════════════

-- Disease graphs (cached results from build_disease_graph)
CREATE TABLE IF NOT EXISTS disease_graphs (
    id              SERIAL PRIMARY KEY,
    disease_name    TEXT NOT NULL,
    disease_id      TEXT,
    graph_json      JSONB NOT NULL,
    parameters      JSONB NOT NULL DEFAULT '{}',
    n_nodes         INTEGER NOT NULL DEFAULT 0,
    n_edges         INTEGER NOT NULL DEFAULT 0,
    sources_queried TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_disease_params UNIQUE (disease_name, parameters)
);

CREATE INDEX IF NOT EXISTS idx_graphs_disease
    ON disease_graphs (disease_name);
CREATE INDEX IF NOT EXISTS idx_graphs_created
    ON disease_graphs (created_at DESC);


-- Pipeline jobs (track execution history)
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id                TEXT PRIMARY KEY,
    disease               TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'pending',
    top_n_targets         INTEGER NOT NULL DEFAULT 5,
    candidates_per_target INTEGER NOT NULL DEFAULT 100,
    result_json           JSONB,
    report_html           TEXT,
    report_path           TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at          TIMESTAMPTZ,
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_disease
    ON pipeline_jobs (disease);
CREATE INDEX IF NOT EXISTS idx_jobs_status
    ON pipeline_jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created
    ON pipeline_jobs (created_at DESC);


-- API response cache (TTL-managed)
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key     TEXT PRIMARY KEY,
    source_name   TEXT NOT NULL,
    query_params  JSONB NOT NULL,
    response_json JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at    TIMESTAMPTZ NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_source
    ON api_cache (source_name);
CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON api_cache (expires_at);


-- Identified causal targets (per job)
CREATE TABLE IF NOT EXISTS causal_targets (
    id                SERIAL PRIMARY KEY,
    job_id            TEXT REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
    gene_name         TEXT NOT NULL,
    protein_name      TEXT NOT NULL,
    uniprot_id        TEXT,
    classification    TEXT NOT NULL,
    causal_confidence REAL NOT NULL,
    causal_effect     REAL NOT NULL,
    robustness_score  REAL NOT NULL,
    druggability_score REAL NOT NULL,
    reasoning         TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_targets_job
    ON causal_targets (job_id);
CREATE INDEX IF NOT EXISTS idx_targets_gene
    ON causal_targets (gene_name);


-- Scored candidates (per job)
CREATE TABLE IF NOT EXISTS scored_candidates (
    id                SERIAL PRIMARY KEY,
    job_id            TEXT REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
    rank              INTEGER NOT NULL,
    smiles            TEXT NOT NULL,
    target_gene       TEXT NOT NULL,
    composite_score   REAL NOT NULL,
    causal_confidence REAL,
    binding_affinity  REAL,
    qed_score         REAL,
    is_drug_like      BOOLEAN DEFAULT FALSE,
    is_novel          BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_job
    ON scored_candidates (job_id);
CREATE INDEX IF NOT EXISTS idx_candidates_rank
    ON scored_candidates (rank);
