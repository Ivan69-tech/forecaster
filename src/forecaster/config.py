from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Base de données
    database_url: str = "postgresql://forecaster:forecaster@localhost:5432/forecaster"

    # API RTE
    rte_client_id: str = ""
    rte_client_secret: str = ""

    # API Open-Meteo (pas de clé requise)
    openmeteo_base_url: str = "https://api.open-meteo.com/v1"

    # Modèles ML
    models_dir: Path = Path("/data/models")

    # Monitoring
    mape_threshold: float = 15.0

    # Logging
    log_level: str = "INFO"


settings = Settings()
