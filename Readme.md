# Forecaster — Service de Prévision SGE (L1)

Service de prévision du **Système de Gestion de l'Énergie**.
Il alimente une base PostgreSQL avec des prévisions de consommation électrique et de
production PV, consommées par le Service d'Optimisation (L2).

---

## Fonctionnel

### Ce que fait ce service

| Prévision | Modèle | Pas | Horizon |
|-----------|--------|-----|---------|
| **Consommation électrique** | LightGBM (`ConsumptionModel`) | 15 min | 48 h |
| **Production PV** | LightGBM (`PVProductionModel`) | 15 min | 48 h |
| **Prix spot RTE** | Fetcher RTE (OAuth2) | 1 h | J+1 |

Le modèle de **consommation** utilise les features suivantes :

- **Temporelles** : heure, jour de semaine, mois (encodage sin/cos cyclique), week-end
- **Lags** : consommation à J-1 et J-7 à la même heure
- **Météo** : température prévue et ses lags J-1, J-7
- **Calendaire** : jours fériés français

Le modèle de **production PV** utilise les features suivantes :

- **Météo** : irradiance GHI (W/m²), nébulosité (%), température (°C) — issues d'Open-Meteo
- **Site** : puissance crête installée (kW)
- **Temporelles** : heure et mois (encodage sin/cos — proxy de la position solaire)

> Pas de lags de production dans le modèle PV : contrairement à la consommation (pilotée par
> les habitudes humaines), la production PV est déterministe — elle dépend de la physique
> instantanée. L'irradiance prévue suffit.

### Flux de données

```
API Open-Meteo (prévision météo) ──┐
                                    ├─→ ConsumptionModel (LightGBM) ──→ forecasts_consommation
PostgreSQL (mesures_reelles) ───────┤
                                    ├─→ PVProductionModel (LightGBM) ──→ forecasts_production_pv
API Open-Meteo (archive météo)  ───┘
                                              ↑
API RTE (prix spots OAuth2)  ──────────────────────────────────────→ forecasts_prix_spot
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
   génère la production PV synthétique, entraîne le `ConsumptionModel` et le `PVProductionModel`,
   génère les premières prévisions 48h (consommation + PV via Open-Meteo), puis s'arrête
3. `forecast-service` — démarre le scheduler APScheduler (jobs récurrents)
4. `grafana` — disponible immédiatement sur **<http://localhost:3000>**

### Visualiser les prévisions (Grafana)

Ouvre **<http://localhost:3000>** — aucun login requis (mode anonyme Admin).

Deux dashboards sont automatiquement provisionnés :

**"Prévisions de Consommation"**

- **Courbe 48h** — prévision LightGBM sur `maintenant → +48h`
- **Stat — Prochaine heure** — puissance moyenne prévue (kW)
- **Stat — Pic 48h** — puissance maximale prévue (kW)
- **Stat — Version modèle** — version + MAPE de validation
- **Tableau** — détail au pas 15 min pour les 6 prochaines heures

**"Prix Spots France"**

- **Courbe 48h** — prix EPEX Spot France en escalier (€/MWh), gradient vert→rouge selon le niveau
- **Stat — Prix heure en cours** — prix du pas horaire actif, coloré par seuil
- **Stat — Prix min 24h** — meilleur prix d'achat sur les prochaines 24h (charge BESS)
- **Stat — Prix max 24h** — meilleur prix de vente sur les prochaines 24h (décharge BESS)
- **Stat — Creux 24h** — heure locale Paris du prix minimum (fenêtre de charge optimale)
- **Tableau** — détail horaire des 24 prochaines heures avec coloration par niveau de prix

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
| `RTE_CLIENT_ID` | *(vide)* | Client ID OAuth2 RTE |
| `RTE_CLIENT_SECRET` | *(vide)* | Client Secret OAuth2 RTE |
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
│   ├── consumption.py    — ConsumptionModel (LightGBM) ✓
│   └── pv_production.py  — PVProductionModel (LightGBM) ✓
├── pipeline/
│   ├── forecast.py       — orchestration prévision ✓ (conso + PV)
│   ├── training.py       — réentraînement LightGBM ✓ (conso + PV)
│   └── monitoring.py     — calcul MAPE + déclenchement réentraînement ✓
├── fetchers/
│   ├── openmeteo.py      — météo Open-Meteo ✓ (fetch_forecast + fetch_historical)
│   └── rte.py            — prix spots RTE ✓ (fetch_spot_prices via OAuth2 Wholesale Market v3)
├── db/
│   ├── models.py         — ORM SQLAlchemy (6 tables)
│   ├── readers.py        — requêtes de lecture DB ✓
│   ├── writers.py        — requêtes d'écriture DB ✓ (conso, PV, prix spots)
│   └── session.py        — engine + SessionLocal
├── exceptions.py         — exceptions métier centralisées
├── scheduler/
│   └── jobs.py           — 5 jobs APScheduler
├── config.py             — Settings Pydantic (.env)
└── main.py               — point d'entrée
```

---

## Points ouverts

| # | Sujet | Impact |
|---|-------|--------|
| 1 | Température dans `_load_training_data()` (conso) hardcodée à 0.0 — à remplacer par `fetch_historical()` Open-Meteo (même pattern que `_load_training_data_pv`). | Amélioration de précision en hiver/été |
| 2 | `run_training()` non multi-site — agrège tous les sites. Ajouter `site_id` en paramètre. | À corriger avant déploiement multi-sites |
| 3 | Modèle PV demo entraîné sur données synthétiques — sera remplacé dès le 1er réentraînement hebdomadaire (dimanche 02h00) avec données réelles Open-Meteo | Précision initiale limitée, s'améliore automatiquement |

---

## Améliorations futures

| Sujet | Priorité | Description |
|-------|----------|-------------|
| **Vacances scolaires** | Basse | `is_school_holiday` est hardcodé à 0 dans `training.py` et `forecast.py`. Intégrer un calendrier (ex : package `vacances-scolaires-france`). |
| **Réentraînement asynchrone** | Basse | `_trigger_retraining()` dans `monitoring.py` est synchrone et bloque le thread de monitoring. Passer en thread séparé ou job APScheduler immédiat. |
| **Alertes / notifications** | Moyenne | Pas de mécanisme d'alerte quand la MAPE est haute ou qu'un job échoue (email, Slack, webhook). |
| **API REST** | Selon besoin | Pas d'API HTTP pour interroger les prévisions — le L2 lit directement la DB. À ajouter si un autre consommateur en a besoin. |
| **Health check** | Moyenne | Pas d'endpoint `/health` pour le monitoring infrastructure (utile pour Docker / Kubernetes). |
| **Lags température J-1/J-7** | Basse | Dans `pipeline/forecast.py`, les lags température sont à 0.0. Intégrer l'archive Open-Meteo pour les calculer. |
| **`base.py` save/load abstraits** | Basse | `BaseForecastModel.save()` et `load()` lèvent `NotImplementedError` alors que les sous-classes les implémentent. Les marquer `@abstractmethod`. |
