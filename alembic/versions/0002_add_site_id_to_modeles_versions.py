"""add site_id to modeles_versions — modèles per-site

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "modeles_versions",
        sa.Column("site_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_modeles_versions_site_id", "modeles_versions", ["site_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_modeles_versions_site_id", table_name="modeles_versions")
    op.drop_column("modeles_versions", "site_id")
