"""Ajoute date_generation_forecasts à trajectoires_optimisees.

Stocke le MAX(date_generation) des trois tables de forecasts au moment
du calcul de la trajectoire. Utilisé par l'optimizer pour détecter la
publication de nouveaux forecasts et invalider son cache.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-06
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
        "trajectoires_optimisees",
        sa.Column("date_generation_forecasts", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trajectoires_optimisees", "date_generation_forecasts")
