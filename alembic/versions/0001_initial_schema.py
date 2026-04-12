"""initial schema — création des 6 tables §3.5

Revision ID: 0001
Revises:
Create Date: 2026-04-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- sites ---
    op.create_table(
        "sites",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("nom", sa.String(128), nullable=False),
        sa.Column("capacite_bess_kwh", sa.Float(), nullable=False),
        sa.Column("p_max_bess_kw", sa.Float(), nullable=False),
        sa.Column("p_pv_peak_kw", sa.Float(), nullable=False),
        sa.Column("p_souscrite_kw", sa.Float(), nullable=False),
        sa.Column("soc_min_pct", sa.Float(), nullable=False, server_default="10.0"),
        sa.Column("soc_max_pct", sa.Float(), nullable=False, server_default="90.0"),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("site_id"),
    )
    op.create_index("ix_sites_site_id", "sites", ["site_id"])

    # --- forecasts_prix_spot ---
    op.create_table(
        "forecasts_prix_spot",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prix_eur_mwh", sa.Float(), nullable=False),
        sa.Column("date_generation", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(64), nullable=False, server_default="RTE"),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecasts_prix_spot_site_id", "forecasts_prix_spot", ["site_id"])
    op.create_index("ix_forecasts_prix_spot_timestamp", "forecasts_prix_spot", ["timestamp"])

    # --- forecasts_consommation ---
    op.create_table(
        "forecasts_consommation",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("puissance_kw", sa.Float(), nullable=False),
        sa.Column("horizon_h", sa.Integer(), nullable=False),
        sa.Column("date_generation", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version_modele", sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecasts_consommation_site_id", "forecasts_consommation", ["site_id"])
    op.create_index(
        "ix_forecasts_consommation_timestamp", "forecasts_consommation", ["timestamp"]
    )

    # --- forecasts_production_pv ---
    op.create_table(
        "forecasts_production_pv",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("puissance_kw", sa.Float(), nullable=False),
        sa.Column("horizon_h", sa.Integer(), nullable=False),
        sa.Column("date_generation", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version_modele", sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_forecasts_production_pv_site_id", "forecasts_production_pv", ["site_id"]
    )
    op.create_index(
        "ix_forecasts_production_pv_timestamp", "forecasts_production_pv", ["timestamp"]
    )

    # --- mesures_reelles ---
    op.create_table(
        "mesures_reelles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("conso_kw", sa.Float(), nullable=False),
        sa.Column("production_pv_kw", sa.Float(), nullable=False),
        sa.Column("soc_kwh", sa.Float(), nullable=False),
        sa.Column("puissance_bess_kw", sa.Float(), nullable=False),
        sa.Column("puissance_pdl_kw", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_mesures_reelles_site_id", "mesures_reelles", ["site_id"])
    op.create_index("ix_mesures_reelles_timestamp", "mesures_reelles", ["timestamp"])

    # --- modeles_versions ---
    op.create_table(
        "modeles_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("type_modele", sa.String(32), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("date_entrainement", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mape_validation", sa.Float(), nullable=True),
        sa.Column("actif", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("chemin_artefact", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("modeles_versions")
    op.drop_table("mesures_reelles")
    op.drop_table("forecasts_production_pv")
    op.drop_table("forecasts_consommation")
    op.drop_table("forecasts_prix_spot")
    op.drop_table("sites")
