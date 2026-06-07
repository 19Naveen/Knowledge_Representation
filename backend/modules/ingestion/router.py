import uuid
from typing import Literal

from core.database import get_db
from core.dependencies import get_current_user
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from modules.ingestion import repository as repo
from modules.ingestion.enums import JobStatus, SourceType
from modules.ingestion.schema_diff import compute_diff
from modules.ingestion.schemas import (
    ColumnDiff,
    CreateIngestionJobRequest,
    DatasetResponse,
    DatasetVersionResponse,
    IngestionJobResponse,
    ResolveSchemaMappingRequest,
    SchemaDiffResponse,
)
from modules.ingestion.service import create_ingestion_job, resolve_schema_mapping
from modules.ingestion.storage.minio_client import upload_staging_file
from modules.ingestion.tasks import run_ingestion_pipeline
from sqlalchemy.orm import Session

router = APIRouter(prefix="/data-ingest", tags=["data-ingest"])


def _job_to_response(job) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        dataset_id=job.dataset_id,
        status=job.status.value,
        source_type=job.source_type.value,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ── Create job from DB source ─────────────────────────────────────────────────


@router.post(
    "/jobs", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def create_job(
    payload: CreateIngestionJobRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    _, job = create_ingestion_job(db, payload)
    run_ingestion_pipeline.delay(str(job.id))
    return _job_to_response(job)


# ── Create job from file upload ───────────────────────────────────────────────


@router.post(
    "/jobs/upload",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job_from_file(
    dataset_id: uuid.UUID = Form(...),
    dataset_name: str = Form(...),
    workspace_id: uuid.UUID = Form(...),
    source_type: Literal["csv", "xlsx", "parquet"] = Form(...),
    file: UploadFile = File(...),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    dataset = repo.get_dataset(db, dataset_id)
    if not dataset:
        dataset = repo.create_dataset(
            db,
            {
                "id": dataset_id,
                "workspace_id": workspace_id,
                "name": dataset_name,
                "description": description,
                "source_type": source_type,
            },
        )

    job = repo.create_job(
        db,
        {
            "dataset_id": dataset.id,
            "status": JobStatus.PENDING,
            "source_type": SourceType(source_type),
            "source_config": None,
        },
    )

    staging_path = f"{dataset.workspace_id}/{dataset.id}/staging/{job.id}/{file.filename}"
    upload_staging_file(file_bytes, staging_path)

    job.staging_path = staging_path
    db.commit()
    db.refresh(job)

    run_ingestion_pipeline.delay(str(job.id))
    return _job_to_response(job)


# ── Get job status ────────────────────────────────────────────────────────────


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    job = repo.get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return _job_to_response(job)


# ── Submit schema resolution ──────────────────────────────────────────────────


@router.post("/jobs/{job_id}/resolve", response_model=IngestionJobResponse)
async def resolve_schema(
    job_id: uuid.UUID,
    payload: ResolveSchemaMappingRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    job = resolve_schema_mapping(db, job_id, payload)
    return _job_to_response(job)


# ── List dataset versions ─────────────────────────────────────────────────────


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    datasets = repo.list_datasets(db, workspace_id)
    result = []
    for ds in datasets:
        versions = repo.list_versions(db, ds.id)
        latest = versions[-1] if versions else None
        result.append(DatasetResponse(
            id=ds.id,
            workspace_id=ds.workspace_id,
            name=ds.name,
            description=ds.description,
            source_type=ds.source_type,
            created_at=ds.created_at,
            version_count=len(versions),
            latest_row_count=latest.row_count if latest else None,
            latest_file_size=latest.file_size if latest else None,
        ))
    return result


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    from modules.ingestion.storage.minio_client import delete_object
    storage_paths = repo.delete_dataset(db, dataset_id, workspace_id)
    if not storage_paths and not repo.get_dataset(db, dataset_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    for path in storage_paths:
        try:
            delete_object(path)
        except Exception:
            pass


@router.get(
    "/datasets/{dataset_id}/versions", response_model=list[DatasetVersionResponse]
)
async def list_versions(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    versions = repo.list_versions(db, dataset_id)
    return [DatasetVersionResponse.model_validate(v) for v in versions]


# ── Get latest schema / diff ──────────────────────────────────────────────────


@router.get("/datasets/{dataset_id}/schema", response_model=SchemaDiffResponse)
async def get_schema_diff(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    versions = repo.list_versions(db, dataset_id)
    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No versions found for dataset",
        )

    if len(versions) == 1:
        return SchemaDiffResponse(
            dataset_id=dataset_id,
            has_diff=False,
            added_columns=[],
            missing_columns=[],
            type_changes=[],
            suggested_mappings=[],
        )

    prev, latest = versions[-2], versions[-1]
    diff = compute_diff(prev.schema, latest.schema)

    return SchemaDiffResponse(
        dataset_id=dataset_id,
        has_diff=diff.has_diff,
        added_columns=diff.added_columns,
        missing_columns=diff.missing_columns,
        type_changes=[
            ColumnDiff(
                column=c["column"],
                old_type=c["old_type"],
                new_type=c["new_type"],
            )
            for c in diff.type_changes
        ],
        suggested_mappings=[
            ColumnDiff(
                column=s["column"],
                suggested_target=s["suggested_target"],
            )
            for s in diff.suggested_mappings
        ],
    )
