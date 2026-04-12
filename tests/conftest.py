"""
Fixtures pytest communes à tous les tests.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from forecaster.db.models import Base, Site

# Base de données SQLite en mémoire pour les tests (pas besoin de PostgreSQL)
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine():
    """Crée un engine SQLite en mémoire et initialise le schéma."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Session:
    """Session de test avec rollback automatique après chaque test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_site(db_session: Session) -> Site:
    """Site fictif pour les tests."""
    site = Site(
        site_id="site-test-01",
        nom="Site de test",
        capacite_bess_kwh=500.0,
        p_max_bess_kw=250.0,
        p_pv_peak_kw=300.0,
        p_souscrite_kw=200.0,
        soc_min_pct=10.0,
        soc_max_pct=90.0,
        latitude=43.6047,   # Toulouse
        longitude=1.4442,
    )
    db_session.add(site)
    db_session.flush()
    return site
