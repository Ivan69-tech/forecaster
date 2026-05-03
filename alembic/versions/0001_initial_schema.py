"""Schéma initial complet — toutes les tables de la plateforme SGE.

Inclut : sites, forecasts_*, mesures_reelles, modeles_versions,
         trajectoires_optimisees, trajectoire_pas, ppc_raw.

Revision ID: 0001
Revises:
Create Date: 2026-05-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # sites — table maître, référencée par toutes les autres
    # -------------------------------------------------------------------------
    op.create_table(
        "sites",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("nom", sa.String(128), nullable=False),
        # Paramètres BESS
        sa.Column("capacite_bess_kwh", sa.Float(), nullable=False),
        sa.Column("p_max_bess_kw", sa.Float(), nullable=False),
        sa.Column("rendement_bess", sa.Float(), nullable=True, server_default="0.92"),
        sa.Column("soc_min_pct", sa.Float(), nullable=False, server_default="10.0"),
        sa.Column("soc_max_pct", sa.Float(), nullable=False, server_default="90.0"),
        # Paramètres PV
        sa.Column("p_pv_peak_kw", sa.Float(), nullable=False),
        # Paramètres PDL / contrat réseau
        sa.Column("p_souscrite_kw", sa.Float(), nullable=False),
        sa.Column("p_max_injection_kw", sa.Float(), nullable=True),
        sa.Column("p_max_soutirage_kw", sa.Float(), nullable=True),
        # Coordonnées géographiques (API météo)
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("site_id"),
    )
    op.create_index("ix_sites_site_id", "sites", ["site_id"])

    # -------------------------------------------------------------------------
    # forecasts_prix_spot — prix spot RTE par pas horaire
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # forecasts_consommation — prévisions de consommation par pas 15 min
    # -------------------------------------------------------------------------
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
    op.create_index("ix_forecasts_consommation_timestamp", "forecasts_consommation", ["timestamp"])

    # -------------------------------------------------------------------------
    # forecasts_production_pv — prévisions de production PV par pas 15 min
    # -------------------------------------------------------------------------
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
    op.create_index("ix_forecasts_production_pv_site_id", "forecasts_production_pv", ["site_id"])
    op.create_index("ix_forecasts_production_pv_timestamp", "forecasts_production_pv", ["timestamp"])

    # -------------------------------------------------------------------------
    # mesures_reelles — mesures terrain agrégées par le PPC (pas ~5 min)
    # Alimentée par DBsync. Utilisée par le forecaster pour l'entraînement.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # modeles_versions — registre des modèles ML entraînés
    # -------------------------------------------------------------------------
    op.create_table(
        "modeles_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=True),
        sa.Column("type_modele", sa.String(32), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("date_entrainement", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mape_validation", sa.Float(), nullable=True),
        sa.Column("actif", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("chemin_artefact", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_modeles_versions_site_id", "modeles_versions", ["site_id"])

    # -------------------------------------------------------------------------
    # trajectoires_optimisees — métadonnées d'une trajectoire BESS calculée
    # Écrite par l'optimizer, lue par le PPC via API.
    # -------------------------------------------------------------------------
    op.create_table(
        "trajectoires_optimisees",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp_calcul", sa.DateTime(timezone=True), nullable=False),
        sa.Column("soe_initial_kwh", sa.Float(), nullable=False),
        sa.Column("statut", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("derive_pct", sa.Float(), nullable=True),
        sa.Column("horizon_debut", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_fin", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_trajectoires_optimisees_site_id", "trajectoires_optimisees", ["site_id"]
    )
    op.create_index(
        "ix_trajectoires_optimisees_timestamp_calcul",
        "trajectoires_optimisees",
        ["timestamp_calcul"],
    )

    # -------------------------------------------------------------------------
    # trajectoire_pas — table glissante (site_id, timestamp) → setpoints BESS
    # À chaque recalcul, les pas futurs sont remplacés. Les pas passés sont conservés.
    # -------------------------------------------------------------------------
    op.create_table(
        "trajectoire_pas",
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("energie_kwh", sa.Float(), nullable=False),
        sa.Column("soe_cible_kwh", sa.Float(), nullable=False),
        sa.Column("insertion_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.site_id"]),
        sa.PrimaryKeyConstraint("site_id", "timestamp"),
    )
    op.create_index(
        "ix_trajectoire_pas_site_timestamp",
        "trajectoire_pas",
        ["site_id", sa.text("timestamp DESC")],
    )

    # -------------------------------------------------------------------------
    # ppc_raw — données brutes terrain (format clé-valeur), toutes sources confondues.
    # Écrite par DBsync. Usage : monitoring, debug, Grafana.
    # Politique de rétention recommandée : 30 jours.
    # -------------------------------------------------------------------------
    op.create_table(
        "ppc_raw",
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("site_id", "key", "timestamp"),
    )
    op.create_index("ix_ppc_raw_site_id", "ppc_raw", ["site_id"])
    op.create_index("ix_ppc_raw_timestamp", "ppc_raw", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_ppc_raw_timestamp", table_name="ppc_raw")
    op.drop_index("ix_ppc_raw_site_id", table_name="ppc_raw")
    op.drop_table("ppc_raw")

    op.drop_index("ix_trajectoire_pas_site_timestamp", table_name="trajectoire_pas")
    op.drop_table("trajectoire_pas")

    op.drop_index("ix_trajectoires_optimisees_timestamp_calcul", table_name="trajectoires_optimisees")
    op.drop_index("ix_trajectoires_optimisees_site_id", table_name="trajectoires_optimisees")
    op.drop_table("trajectoires_optimisees")

    op.drop_index("ix_modeles_versions_site_id", table_name="modeles_versions")
    op.drop_table("modeles_versions")

    op.drop_index("ix_mesures_reelles_timestamp", table_name="mesures_reelles")
    op.drop_index("ix_mesures_reelles_site_id", table_name="mesures_reelles")
    op.drop_table("mesures_reelles")

    op.drop_index("ix_forecasts_production_pv_timestamp", table_name="forecasts_production_pv")
    op.drop_index("ix_forecasts_production_pv_site_id", table_name="forecasts_production_pv")
    op.drop_table("forecasts_production_pv")

    op.drop_index("ix_forecasts_consommation_timestamp", table_name="forecasts_consommation")
    op.drop_index("ix_forecasts_consommation_site_id", table_name="forecasts_consommation")
    op.drop_table("forecasts_consommation")

    op.drop_index("ix_forecasts_prix_spot_timestamp", table_name="forecasts_prix_spot")
    op.drop_index("ix_forecasts_prix_spot_site_id", table_name="forecasts_prix_spot")
    op.drop_table("forecasts_prix_spot")

    op.drop_index("ix_sites_site_id", table_name="sites")
    op.drop_table("sites")
