# Plan: Data Import Wizard (DataForge-driven, review-gated on every import)

## Context

The schema-diff resolver (map/drop/accept inside Data Import) is too thin: it only appears on a
diff, can't clean a *first* import (e.g. junk `Unnamed: N` columns from a title row), and gives the
user no real control before a version is written. Decision: make **every import review-gated** and
do the actual shaping in **DataForge** (the Data Transform page), which becomes the single surface
for transforming + committing an incoming dataset.

New flow:

```
Upload / Connect  →  stage source + create PENDING job (NO auto-pipeline)
        ↓ redirect
DataForge (import mode):  preview sample rows + inferred schema
        + (if prior version) schema diff & suggested renames
        + user builds a transform plan: drop / rename / cast / filter / fill-missing
        ↓ Save & Apply
POST /commit (transform plan)  →  Celery pipeline applies transforms → writes new version
```

Downstream (EDA/Query/Transform-preview) keep reading the latest version, unchanged.

**Decisions (confirmed):** wizard on *every* import; transforms = drop/rename/cast **+ filter +
fill-missing**; UI = **hand off to DataForge** (not an inline Data Import stepper).

This supersedes the ad-hoc resolve UI in `DataImportPage` and the test tasks T2–T5 (their intent
is folded into B4). T1 (pytest infra + schema_diff tests, commit `422de80`) stands.

---

## Backend

### B1 — Transform model + engine (pure, TDD)
New `backend/modules/ingestion/transforms.py`.
- Pydantic `TransformStep` (discriminated by `type`): `drop{column}`, `rename{column,to}`,
  `cast{column,to_type}` (to_type ∈ string/integer/decimal/boolean/timestamp),
  `filter{column,op,value}` (op ∈ eq/ne/lt/le/gt/ge/isnull/notnull),
  `fillna{column,strategy,value?}` (strategy ∈ value/mean/median/mode).
- `apply_transforms(df, steps) -> df`: applies steps in order. Reuse the existing
  map/drop/cast semantics from `schema_diff.apply_rules` (CAST_TYPE_MAP; timestamp via
  `pd.to_datetime(errors="coerce")`). filter builds a boolean mask; fillna uses the strategy
  (mean/median only for numeric; mode/value general). Unknown/absent column → safe no-op.
- Add `TransformType.FILTER`, `TransformType.FILLNA` to `enums.py` (string values). These live in
  JSON on the job, NOT a Postgres enum column, so **no DB migration** is required.
- Unit tests in `tests/test_transforms.py` cover every step type + ordering + no-op safety.

### B2 — Staged-preview endpoint
`GET /data-ingest/jobs/{id}/staged-preview?limit=N` (default 50).
- Load the staged source via the existing connector (`_load_source` in `tasks.py` — refactor the
  source-dispatch out of the Celery task into a reusable helper so the endpoint can call it),
  `head(N)`, run `infer_schema`.
- Return `{columns, dataset_schema, sample_rows, previous_schema|null, diff|null}` where `diff` =
  `compute_diff(latest_version.schema, inferred)` (with `suggested_mappings`) when a prior version
  exists. Reuse `schema_inference.infer_schema` and `schema_diff.compute_diff`.
- Bound cost: read a sample only (extend CSV/XLSX/Parquet connectors with an optional `nrows`, or
  read then slice for v1 — note the limitation).

### B3 — Defer dispatch + commit endpoint
- `create_job` and `create_job_from_file` (`router.py`/`service.py`): stage / store config and
  create the PENDING job but **do not** call `run_ingestion_pipeline.delay(...)`. Return the job so
  the frontend can hand off to DataForge. (PENDING now means "awaiting review".)
- New `POST /data-ingest/jobs/{id}/commit` with body `{transforms: TransformStep[]}`:
  validate job is PENDING, persist the plan on `job.source_config["_transforms"]`, then
  `run_ingestion_pipeline.delay(...)`.
- `run_ingestion_pipeline`: load source → `apply_transforms(df, job._transforms)` →
  `infer_schema(result)` → write Parquet → create `DatasetVersion` → SUCCESS → delete staging.
  Remove the old auto-pause-on-diff branch (review happens in DataForge now). Keep the
  `accept_new_schema`/MappingRule code paths only if still referenced; otherwise delete them and the
  now-dead `/resolve` endpoint + `store_pending_schema`.
- Update `tests/` (schema-diff endpoint behavior, commit dispatch, pipeline applies transforms).

### B4 — Backend tests (folds in T2–T4)
pytest coverage for B1 (engine), B2 (preview/diff shape, mocked connector+repo), B3 (commit
persists plan + dispatches; pipeline applies transforms and versions). Mock repo/SessionLocal/MinIO
with `SimpleNamespace`/`MagicMock` (no DB) per the T1 conftest pattern.

---

## Frontend

### F1 — Import becomes a hand-off
`features/data-import/DataImportPage.tsx`: upload/connect call the (non-dispatching) endpoints, then
`navigate("/app/data-transform", { state: { jobId, datasetId, datasetName } })`. Remove the
polling / schema_diff / resolve UI (superseded). Keep the dataset list + add-data entry points.

### F2 — DataForge import-review mode
`features/data-transform/DataTransformPage.tsx`: when arriving with a `jobId` (router state),
enter import mode:
- Fetch `/jobs/{id}/staged-preview`; render sample rows + schema in the existing table; if `diff`
  present, surface added/missing/type-changes and one-click "rename to <previous>" suggestions.
- Let the user build a transform plan with the existing step UI, extended to the real op set
  (drop/rename/cast/filter/fill-missing) bound to real columns.
- "Save & Apply" → `POST /jobs/{id}/commit {transforms}` → poll `/jobs/{id}` to SUCCESS/FAILED →
  on success navigate back to Data Import (or dataset detail). Use the `errMessage` helper for
  readable errors.
Normal (no-jobId) visits keep the Phase-3 read-only preview behavior.

### F3 — Step builder ↔ real columns + new ops
Wire the DataForge step builder to the staged columns and the new transform op set so each step maps
1:1 to a backend `TransformStep`. (May merge with F2.)

---

## Docs (per CLAUDE.md, same PR as code)
- `backend/README.md`: new endpoints (`/staged-preview`, `/commit`), the review-gated flow, the
  transform model + `TransformType.FILTER/FILLNA`, removal of auto-dispatch/`/resolve`; changelog.
- `frontend/README.md`: import hand-off + DataForge import mode; changelog.
- `README.md`: changelog note on review-gated ingestion.

## Critical files
- Backend: `modules/ingestion/{transforms.py(new),enums.py,tasks.py,service.py,router.py,schemas.py}`,
  `connectors/*` (sample read), `tests/*`.
- Frontend: `features/data-import/DataImportPage.tsx`, `features/data-transform/DataTransformPage.tsx`,
  `lib/hooks/useDatasets.ts` (commit/preview helpers), `lib/http.ts` (errMessage, from T5).

## Task order / dependencies
B1 → B2 → B3 (backend, sequential; B4 alongside) → F1 → F2/F3 → Docs. F2 depends on B2+B3.

## Verification
1. Start stack (`make backend` + Celery worker) and frontend.
2. Upload a CSV with a junk leading column → redirected to DataForge → drop that column, add a
   `filter`/`fillna` step → Save & Apply → job SUCCESS → new version's schema reflects the
   transforms (verify via dataset detail panel + `/query/.../preview`).
3. Upload a differently-shaped CSV to the same dataset → DataForge shows the diff + suggested
   renames → map to previous names → commit → version increments, schema matches the mapping.
4. `uv run pytest -q` green (B1–B4); `npm run build` clean.
