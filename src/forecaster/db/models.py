"""
Modèles ORM SQLAlchemy — tables §3.5 des spécifications techniques SGE.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Site(Base):
    """
    Table `sites` — paramètres techniques par site (§3.5).
    """

    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    nom: Mapped[str] = mapped_column(String(128), nullable=False)
    capacite_bess_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    p_max_bess_kw: Mapped[float] = mapped_column(Float, nullable=False)
    p_pv_peak_kw: Mapped[float] = mapped_column(Float, nullable=False)
    p_souscrite_kw: Mapped[float] = mapped_column(Float, nullable=False)
    soc_min_pct: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    soc_max_pct: Mapped[float] = mapped_column(Float, nullable=False, default=90.0)
    # Coordonnées géographiques pour l'API météo
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    spot_prices: Mapped[list["SpotPriceForecast"]] = relationship(back_populates="site")
    conso_forecasts: Mapped[list["ConsumptionForecast"]] = relationship(back_populates="site")
    pv_forecasts: Mapped[list["PVProductionForecast"]] = relationship(back_populates="site")
    real_measures: Mapped[list["RealMeasure"]] = relationship(back_populates="site")


class SpotPriceForecast(Base):
    """
    Table `forecasts_prix_spot` — prix spots RTE par pas horaire (§3.5).
    """

    __tablename__ = "forecasts_prix_spot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sites.site_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prix_eur_mwh: Mapped[float] = mapped_column(Float, nullable=False)
    date_generation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="RTE")

    site: Mapped["Site"] = relationship(back_populates="spot_prices")


class ConsumptionForecast(Base):
    """
    Table `forecasts_consommation` — prévisions de consommation par pas 15 min (§3.5).
    """

    __tablename__ = "forecasts_consommation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sites.site_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    puissance_kw: Mapped[float] = mapped_column(Float, nullable=False)
    horizon_h: Mapped[int] = mapped_column(Integer, nullable=False)  # horizon en heures (ex. 48)
    date_generation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    version_modele: Mapped[str] = mapped_column(String(32), nullable=False)

    site: Mapped["Site"] = relationship(back_populates="conso_forecasts")


class PVProductionForecast(Base):
    """
    Table `forecasts_production_pv` — prévisions de production PV par pas 15 min (§3.5).
    """

    __tablename__ = "forecasts_production_pv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sites.site_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    puissance_kw: Mapped[float] = mapped_column(Float, nullable=False)
    horizon_h: Mapped[int] = mapped_column(Integer, nullable=False)
    date_generation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    version_modele: Mapped[str] = mapped_column(String(32), nullable=False)

    site: Mapped["Site"] = relationship(back_populates="pv_forecasts")


class RealMeasure(Base):
    """
    Table `mesures_reelles` — mesures terrain remontées par le contrôleur (§3.5).
    Pas 5 min — source : Logiciel 3 (Contrôleur Local).
    """

    __tablename__ = "mesures_reelles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sites.site_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    conso_kw: Mapped[float] = mapped_column(Float, nullable=False)
    production_pv_kw: Mapped[float] = mapped_column(Float, nullable=False)
    soc_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    puissance_bess_kw: Mapped[float] = mapped_column(Float, nullable=False)
    puissance_pdl_kw: Mapped[float] = mapped_column(Float, nullable=False)

    site: Mapped["Site"] = relationship(back_populates="real_measures")


class ModelVersion(Base):
    """
    Table `modeles_versions` — registre des modèles entraînés (§3.5).
    """

    __tablename__ = "modeles_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type_modele: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "consumption" | "pv_production"
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    date_entrainement: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    mape_validation: Mapped[float | None] = mapped_column(Float, nullable=True)
    actif: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    chemin_artefact: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
