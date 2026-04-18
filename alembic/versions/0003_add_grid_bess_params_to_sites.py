"""add p_max_injection_kw / p_max_soutirage_kw / rendement_bess to sites

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-18

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sites",
        sa.Column("p_max_injection_kw", sa.Float, nullable=True),
    )
    op.add_column(
        "sites",
        sa.Column("p_max_soutirage_kw", sa.Float, nullable=True),
    )
    op.add_column(
        "sites",
        sa.Column("rendement_bess", sa.Float, nullable=True, server_default="0.92"),
    )


def downgrade() -> None:
    op.drop_column("sites", "rendement_bess")
    op.drop_column("sites", "p_max_soutirage_kw")
    op.drop_column("sites", "p_max_injection_kw")
