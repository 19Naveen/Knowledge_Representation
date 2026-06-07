import uuid

from sqlalchemy.orm import Session

from modules.workspace.models import Workspace


def create_workspace(db: Session, owner_id: str, name: str, description: str | None) -> Workspace:
    slug = name.lower().replace(" ", "-")
    ws = Workspace(name=name, slug=slug, description=description, owner_id=owner_id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def list_workspaces(db: Session, owner_id: str) -> list[Workspace]:
    return db.query(Workspace).filter(Workspace.owner_id == owner_id).order_by(Workspace.created_at).all()


def get_workspace(db: Session, workspace_id: uuid.UUID, owner_id: str) -> Workspace | None:
    return db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.owner_id == owner_id).first()


def delete_workspace(db: Session, workspace_id: uuid.UUID, owner_id: str) -> bool:
    ws = get_workspace(db, workspace_id, owner_id)
    if not ws:
        return False
    db.delete(ws)
    db.commit()
    return True
