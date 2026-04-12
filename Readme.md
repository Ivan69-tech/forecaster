# Forecaster — Service de Prévision SGE (L1)

Service de prévision du **Système de Gestion de l'Énergie (SGE)** de Tewa Solar.
Il alimente une base PostgreSQL avec des prévisions de consommation électrique et de
production PV, consommées par le Service d'Optimisation (L2).

---

## Fonctionnel

### Ce que fait ce service

| Prévision | Modèle | Pas | Horizon |
|-----------|--------|-----|---------|
| **Consommation électrique** | LightGBM (`ConsumptionModel`) | 15 min | 48 h |
| Production PV *(à venir)* | LightGBM (`PVProductionModel`) | 15 min | 48 h |
| Prix spot RTE *(à venir)* | Fetcher RTE | 1 h | J+1 |

Le modèle de consommation utilise les features suivantes :
- **Temporelles** : heure, jour de semaine, mois (encodage sin/cos cyclique)
- **Lags** : consommation à J-1 et J-7 à la même heure
- **Météo** : température prévue et ses lags J-1, J-7
- **Calendaire** : jours fériés français, week-end

### Flux de données

```
API Open-Meteo (météo)  ──┐
PostgreSQL (mesures_reelles) ─┤─→ ConsumptionModel (LightGBM) ──→ forecasts_consommation
                             │
API RTE (prix spots)  ───────┘                      ↑
                                          scheduler/jobs.py (APScheduler)
mesures_reelles ──→ pipeline/monitoring.py ──→ pipeline/training.py
                      (MAPE > 15% → réentraînement)
```

### Jobs planifiés (fuseau Europe/Paris)

| Heure | Fréquence | Job | Fonction |
|-------|-----------|-----|----------|
| 06h00 | Quotidien | Prévisions 48h (J + J+1) | `run_forecast_all_sites(horizon_h=48)` |
| 16h00 | Quotidien | Prix spots RTE J+1 | `fetch_spot_prices(tomorrow)` |
| 12h00, 18h00, 00h00 | Quotidien | Prévisions intraday 24h | `run_forecast_all_sites(horizon_h=24)` |
| 02h00 | Dimanche | Réentraînement LightGBM | `run_training_all()` |
| — | Toutes les heures | Monitoring MAPE | `check_mape_all_sites()` → réentraînement si MAPE > seuil |

### Tables PostgreSQL

| Table | Rôle |
|-------|------|
| `sites` | Paramètres techniques par site (capacité BESS, puissance PV, coordonnées GPS…) |
| `forecasts_consommation` | Prévisions de conso, pas 15 min, horizon 48 h |
| `forecasts_production_pv` | Prévisions PV, pas 15 min, horizon 48 h |
| `forecasts_prix_spot` | Prix spots RTE, pas horaire |
| `mesures_reelles` | Mesures terrain (Contrôleur Local L3), pas 5 min |
| `modeles_versions` | Registre des artefacts LightGBM (un seul `actif=True` par type) |

---

## Déploiement local (Docker)

### Prérequis

- Docker + Docker Compose
- Pas de PostgreSQL local requis (inclus dans le compose)

### Lancement

```bash
# 1. Copier les variables d'environnement
cp .env.example .env

# 2. Démarrer l'ensemble des services
docker compose up -d

# 3. Suivre l'initialisation (entraînement du modèle, ~1-2 min)
docker compose logs -f forecast-init
```

**Ordre de démarrage automatique :**

1. `postgres` — démarre et passe le healthcheck
2. `forecast-init` — insère un site démo, charge 100 jours d'historique synthétique,
   entraîne le `ConsumptionModel`, génère la première prévision 48h, puis s'arrête
3. `forecast-service` — démarre le scheduler APScheduler (jobs récurrents)
4. `grafana` — disponible immédiatement sur **http://localhost:3000**

### Visualiser les prévisions (Grafana)

Ouvre **http://localhost:3000** — aucun login requis (mode anonyme Admin).

Le dashboard **"Prévisions de Consommation"** est automatiquement provisionné :

- **Courbe 48h** — prévision LightGBM sur `maintenant → +48h`
- **Stat — Prochaine heure** — puissance moyenne prévue (kW)
- **Stat — Pic 48h** — puissance maximale prévue (kW)
- **Stat — Version modèle** — version + MAPE de validation
- **Tableau** — détail au pas 15 min pour les 6 prochaines heures

### Commandes utiles

```bash
# Logs de l'entraînement initial
docker compose logs forecast-init

# Suivre le scheduler en temps réel
docker compose logs -f forecast-service

# Régénérer les prévisions (rejoue le script d'init)
docker compose run --rm forecast-init python scripts/init_demo.py

# Appliquer les migrations manuellement (si besoin)
docker compose exec forecast-service alembic upgrade head

# Arrêt propre
docker compose down

# Tout effacer (volumes compris — repart de zéro)
docker compose down -v
```

---

## Développement local (sans Docker)

### Prérequis

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — gestionnaire de packages
- PostgreSQL 16 accessible sur `localhost:5432` (ou via `docker compose up postgres`)

### Installation

```bash
uv pip install -e ".[dev]"
```

### Variables d'environnement

Copier `.env.example` → `.env` et renseigner les valeurs :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATABASE_URL` | `postgresql://forecaster:forecaster@localhost:5432/forecaster` | URL de connexion PostgreSQL |
| `RTE_API_TOKEN` | *(vide)* | Token OAuth2 RTE (non requis si le fetcher n'est pas utilisé) |
| `OPENMETEO_BASE_URL` | `https://api.open-meteo.com/v1` | URL de base Open-Meteo (pas de clé requise) |
| `MODELS_DIR` | `/data/models` | Répertoire de stockage des artefacts LightGBM |
| `MAPE_THRESHOLD` | `15.0` | Seuil MAPE (%) déclenchant un réentraînement automatique |
| `LOG_LEVEL` | `INFO` | Niveau de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Migrations

```bash
# Appliquer les migrations
uv run alembic upgrade head

# Générer une nouvelle migration après modification des modèles ORM
uv run alembic revision --autogenerate -m "description_courte"
```

### Lancer le service

```bash
uv run python -m forecaster.main
```

### Données de test synthétiques

Le repo inclut un générateur de données d'un an au pas 15 min
représentatives d'un site industriel 2×8h (~700 kW peak) :

```bash
# Régénérer le CSV (reproductible, seed=42)
uv run python tests/fixtures/generate_load_data.py
# → tests/fixtures/load_history_2025.csv  (35 040 lignes)
```

---

## Tests

```bash
# Suite complète
uv run pytest

# Sans couverture (plus rapide)
uv run pytest --no-cov

# Un test spécifique
uv run pytest tests/test_predictors/test_consumption_model.py::test_train_retourne_mape_inferieure_au_seuil

# Lint et formatage
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

Les tests utilisent **SQLite en mémoire** — aucune connexion PostgreSQL requise.
Les appels HTTP sont mockés. Convention de chemin miroir :
`pipeline/forecast.py` → `tests/pipeline/test_forecast.py`.

---

## Architecture

```
src/forecaster/
├── predictors/
│   ├── base.py           — interface BaseForecastModel + ForecastPoint
│   ├── consumption.py    — ConsumptionModel (LightGBM) ✓ implémenté
│   └── pv_production.py  — PVProductionModel (LightGBM) — stub
├── pipeline/
│   ├── forecast.py       — orchestration prévision (stub)
│   ├── training.py       — réentraînement LightGBM ✓ implémenté (conso)
│   └── monitoring.py     — calcul MAPE + déclenchement réentraînement (stub)
├── fetchers/
│   ├── openmeteo.py      — météo Open-Meteo (stub)
│   └── rte.py            — prix spots RTE OAuth2 (stub)
├── db/
│   ├── models.py         — ORM SQLAlchemy (6 tables)
│   ├── readers.py        — requêtes de lecture DB
│   └── session.py        — engine + SessionLocal
├── scheduler/
│   └── jobs.py           — 5 jobs APScheduler
├── config.py             — Settings Pydantic (.env)
└── main.py               — point d'entrée
```

---

## Points ouverts

| # | Sujet | Impact |
|---|-------|--------|
| 1 | `pipeline/forecast.py` non implémenté — la prévision en prod passe par `scripts/init_demo.py` | Bloquant pour la mise en production |
| 2 | `fetchers/openmeteo.py` non implémenté — température = 0 dans l'entraînement actuel | Dégrade la précision du modèle en hiver/été |
| 3 | `fetchers/rte.py` non implémenté — prix spots indisponibles | Bloquant pour l'optimisation L2 |
| 4 | `PVProductionModel` non implémenté | Bloquant pour les sites avec panneaux PV |
| 5 | `pipeline/monitoring.py` non implémenté — pas de réentraînement automatique sur dérive MAPE | Dégradation silencieuse en production |
| 6 | Température dans `_load_training_data()` = 0 — à remplacer par l'archive Open-Meteo | Amélioration de précision |
| 7 | `run_training()` non multi-site — agrège tous les sites | À corriger avant déploiement multi-sites |
