import uuid

from celery_app import celery_app
from core.database import SessionLocal
from modules.ingestion import repository as repo
from modules.ingestion.enums import JobStatus
from modules.ingestion.schema_inference import infer_schema
from modules.ingestion.schema_diff import compute_diff, apply_rules
from modules.ingestion.source_loader import load_source
from modules.ingestion.storage.minio_client import upload_dataframe_as_parquet, delete_object


@celery_app.task(bind=True, max_retries=0, name="ingestion.run_pipeline")
def run_ingestion_pipeline(self, job_id: str):
    db = SessionLocal()
    try:
        job = repo.get_job(db, uuid.UUID(job_id))
        if not job:
            return {"error": f"Job {job_id} not found"}

        # Capture staging_path before any DB refresh so delete still works
        staging_path = job.staging_path

        repo.update_job_status(db, job.id, JobStatus.RUNNING)
        db.refresh(job)

        dataset = repo.get_dataset(db, job.dataset_id)

        df = load_source(job)
        inferred_schema = infer_schema(df)

        latest_version = repo.get_latest_version(db, job.dataset_id)
        if latest_version:
            diff = compute_diff(latest_version.schema, inferred_schema)
            existing_rules = repo.get_mapping_rules(db, job.dataset_id)
            accept_new_schema = bool((job.source_config or {}).get("_accept_new_schema"))

            if diff.has_diff and not existing_rules and not accept_new_schema:
                repo.store_pending_schema(db, job.id, inferred_schema)
                return {"status": "awaiting_schema_resolution", "job_id": job_id}

            # When accepting the new schema as-is, skip any prior mapping rules.
            if existing_rules and not accept_new_schema:
                df = apply_rules(df, existing_rules)
                inferred_schema = infer_schema(df)

        next_version = (latest_version.version + 1) if latest_version else 1

        # Storage layout (inside bucket 'datasets'): {workspace_id}/{dataset_id}/raw/v{n}/data.parquet
        storage_path = f"{dataset.workspace_id}/{dataset.id}/raw/v{next_version}/data.parquet"

        file_size = upload_dataframe_as_parquet(df, storage_path)

        repo.create_dataset_version(db, {
            "dataset_id": job.dataset_id,
            "version": next_version,
            "storage_path": storage_path,
            "row_count": len(df),
            "column_count": len(df.columns),
            "schema": inferred_schema,
            "file_size": file_size,
        })

        repo.update_job_status(db, job.id, JobStatus.SUCCESS)

        # Delete staging file after marking SUCCESS so the path is still known
        if staging_path:
            try:
                delete_object(staging_path)
            except Exception:
                pass

        return {"status": "SUCCESS", "storage_path": storage_path, "version": next_version}

    except Exception as exc:
        repo.update_job_status(db, job.id, JobStatus.FAILED, error_message=str(exc))
        raise
    finally:
        db.close()
