import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from modules.ingestion.enums import JobStatus, SourceType, TransformType
from modules.ingestion.models import Dataset, DatasetVersion, IngestionJob, MappingRule


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset(db: Session, payload: dict) -> Dataset:
    dataset = Dataset(**payload)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(db: Session, dataset_id: uuid.UUID) -> Dataset | None:
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def list_datasets(db: Session, workspace_id: uuid.UUID) -> list[Dataset]:
    return (
        db.query(Dataset)
        .filter(Dataset.workspace_id == workspace_id)
        .order_by(Dataset.created_at.desc())
        .all()
    )


def delete_dataset(db: Session, dataset_id: uuid.UUID, workspace_id: uuid.UUID) -> list[str]:
    """Delete dataset + all child records. Returns storage_paths so caller can clean MinIO."""
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.workspace_id == workspace_id)
        .first()
    )
    if not dataset:
        return []
    versions = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).all()
    storage_paths = [v.storage_path for v in versions]
    db.query(MappingRule).filter(MappingRule.dataset_id == dataset_id).delete()
    db.query(IngestionJob).filter(IngestionJob.dataset_id == dataset_id).delete()
    db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).delete()
    db.delete(dataset)
    db.commit()
    return storage_paths


# ── IngestionJob ──────────────────────────────────────────────────────────────

def create_job(db: Session, payload: dict) -> IngestionJob:
    job = IngestionJob(**payload)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: uuid.UUID) -> IngestionJob | None:
    return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()


def update_job_status(
    db: Session,
    job_id: uuid.UUID,
    status: JobStatus,
    error_message: str | None = None,
) -> None:
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if job:
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        if error_message is not None:
            job.error_message = error_message
        db.commit()


def store_pending_schema(db: Session, job_id: uuid.UUID, schema: dict) -> None:
    """Store the inferred schema on the job's source_config while awaiting resolution."""
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if job:
        config = dict(job.source_config or {})
        config["_pending_schema"] = schema
        job.source_config = config
        job.status = JobStatus.PENDING
        job.updated_at = datetime.now(timezone.utc)
        db.commit()


def get_pending_schema(db: Session, job_id: uuid.UUID) -> dict | None:
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if job and job.source_config:
        return job.source_config.get("_pending_schema")
    return None


# ── DatasetVersion ────────────────────────────────────────────────────────────

def create_dataset_version(db: Session, payload: dict) -> DatasetVersion:
    version = DatasetVersion(**payload)
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def get_latest_version(db: Session, dataset_id: uuid.UUID) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.desc())
        .first()
    )


def list_versions(db: Session, dataset_id: uuid.UUID) -> list[DatasetVersion]:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.asc())
        .all()
    )


# ── MappingRule ───────────────────────────────────────────────────────────────

def upsert_mapping_rules(db: Session, dataset_id: uuid.UUID, rules: list[dict]) -> list[MappingRule]:
    """Replace all mapping rules for a dataset atomically."""
    db.query(MappingRule).filter(MappingRule.dataset_id == dataset_id).delete()
    new_rules = [MappingRule(dataset_id=dataset_id, **rule) for rule in rules]
    db.add_all(new_rules)
    db.commit()
    for rule in new_rules:
        db.refresh(rule)
    return new_rules


def get_mapping_rules(db: Session, dataset_id: uuid.UUID) -> list[MappingRule]:
    return db.query(MappingRule).filter(MappingRule.dataset_id == dataset_id).all()
