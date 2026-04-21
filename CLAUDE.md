# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commandes essentielles

```bash
# Installer les dépendances (inclut les outils dev)
uv pip install -e ".[dev]"

# Lancer tous les tests
uv run pytest

# Lancer un test unique
uv run pytest tests/test_fetchers/test_rte.py::test_spot_price_row_dataclass

# Lancer les tests sans couverture (plus rapide)
uv run pytest --no-cov

# Lint et formatage
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Appliquer les migrations en base
uv run alembic upgrade head

# Générer une nouvelle migration après modification des modèles ORM
uv run alembic revision --autogenerate -m "description_courte"

# Lancer le service localement (nécessite une DB PostgreSQL)
uv run python -m forecaster.main

# Lancer avec Docker (DB incluse)
docker compose up -d
docker compose logs -f forecast-service
```

## Architecture

Ce repo implémente le **Logiciel 1 — Service de Prévision**. Il alimente une base PostgreSQL avec des prévisions de prix spots, de consommation et de production PV, consommées par le Service d'Optimisation (L2, repo séparé).

### Flux de données

```
API RTE (prix spots)  ──┐
API Open-Meteo (météo) ──┤─→ fetchers/ ──→ pipeline/forecast.py ──→ PostgreSQL
PostgreSQL (historique) ─┘                          ↑
                                          scheduler/jobs.py (APScheduler)
PostgreSQL (mesures_reelles) ──→ pipeline/monitoring.py ──→ pipeline/training.py
                                  (MAPE > 15% → réentraînement)
```

### Couches et responsabilités

| Couche | Modules | Rôle |
|--------|---------|------|
| **Entrée** | `fetchers/rte.py`, `fetchers/openmeteo.py` | Appels HTTP vers les APIs externes, retournent des dataclasses typées |
| **Modèles ML** | `predictors/base.py`, `predictors/consumption.py`, `predictors/pv_production.py` | LightGBM — `BaseForecastModel` définit l'interface (`train`, `predict`, `save`, `load`) |
| **Pipeline** | `pipeline/forecast.py`, `pipeline/training.py`, `pipeline/monitoring.py` | Orchestration : assemble les fetchers, les modèles et les couches DB |
| **DB** | `db/models.py`, `db/session.py` | ORM SQLAlchemy + factory de sessions. Les requêtes métier vont dans `db/readers.py` et `db/writers.py` (à créer) |
| **Scheduling** | `scheduler/jobs.py` | Définit les 7 jobs APScheduler — appelle uniquement des fonctions de `pipeline/` |
| **Config** | `config.yaml` | Tous les paramètres configurables (jamais de valeurs en dur dans le code) |

### Schéma de données (6 tables PostgreSQL)

- `sites` — paramètres techniques par site (capacité BESS, puissance PV, coordonnées GPS…)
- `forecasts_prix_spot` — prix spots RTE, pas horaire
- `forecasts_consommation` — prévisions conso, pas 15 min, horizon 48 h
- `forecasts_production_pv` — prévisions PV, pas 15 min, horizon 48 h
- `mesures_reelles` — mesures terrain remontées par le Contrôleur Local (L3), pas 5 min
- `modeles_versions` — registre des artefacts LightGBM entraînés (un seul `actif=True` par type)

### Jobs scheduler (Europe/Paris)

| Heure | Job | Fonction pipeline |
|-------|-----|-------------------|
| 06h00 quotidien | Prévisions 48h (J + J+1) | `run_forecast_all_sites(horizon_h=48)` |
| 16h00 quotidien | Prix spots RTE J+1 | `fetch_spot_prices(tomorrow)` |
| 12h00, 18h00, 00h00 | Prévisions intraday 24h | `run_forecast_all_sites(horizon_h=24)` |
| Dimanche 02h00 | Réentraînement LightGBM | `run_training_all()` |
| Toutes les heures | Monitoring MAPE | `check_mape_all_sites()` → trigger retraining si MAPE > seuil |

### Déploiement

- **Dev local** : `docker compose up` démarre `forecast-service` + `postgres:16`. La DB est exposée sur `localhost:5432`.
- **Production** : `docker compose` sur serveur cloud. Les deux services (L1 et L2) partagent la même instance PostgreSQL existante.
- Les migrations Alembic sont dans `alembic/versions/` — les appliquer manuellement (`alembic upgrade head`) avant chaque déploiement.

### Tests

Les tests utilisent SQLite en mémoire (fixture `test_engine` dans `conftest.py`) — aucune connexion PostgreSQL requise. Les appels HTTP sont mockés. Le miroir de chemin est obligatoire : `pipeline/forecast.py` → `tests/pipeline/test_forecast.py`.

---

## Philosophie générale

- Écris du code simple et explicite. Un développeur junior doit pouvoir comprendre
  chaque fonction sans contexte supplémentaire.
- Préfère la clarté à la concision. Pas de one-liners cryptiques.
- Une fonction = une responsabilité. Si tu dois écrire "et" pour décrire ce que
  fait une fonction, elle fait trop de choses.

## Structure et organisation

- Respecte la structure de répertoires définie dans le README (sources/, models/,
  pipeline/, db/, monitoring/).
- Chaque nouveau module doit avoir un docstring de fichier expliquant son rôle
  en 2-3 lignes.
- Pas de logique métier dans main.py ou scheduler.py — uniquement du câblage.
- Les paramètres configurables vont dans config.yaml jamais en dur
  dans le code.

## Nommage

- Français pour les noms de variables métier (site_id, puissance_kw, soc_kwh,
  horizon_h). Anglais pour la structure technique (class, method, exception names).
- Pas d'abréviations ambiguës. `consumption_model` plutôt que `cm`.
  `forecast_timestamp` plutôt que `ts`.
- Les fonctions qui écrivent en base commencent par `write_`. Celles qui lisent
  commencent par `get_` ou `fetch_`. Celles qui calculent commencent par
  `compute_`.

## Types et interfaces

- Type hints obligatoires sur toutes les fonctions publiques (paramètres + retour).
- Utilise des dataclasses ou Pydantic models pour tous les objets échangés entre
  modules. Pas de dicts non typés qui traversent les couches.
- Les valeurs de retour ambiguës (succès/échec) utilisent un type explicite,
  pas None silencieux.

## Gestion des erreurs

- Pas de `except Exception` silencieux. Chaque exception attrapée doit être
  loggée avec le contexte (site_id, timestamp, valeurs en cause).
- Les erreurs d'API externe (RTE, OpenMeteo) sont retriées 3 fois avec backoff
  exponentiel avant de lever une exception.
- Un site en erreur ne doit jamais bloquer le traitement des autres sites.
  Utilise try/except dans les boucles multi-sites et continue.
- Définis des exceptions métier explicites dans un module exceptions.py
  (ForecastUnavailableError, ModelNotFoundError, etc.).

## Base de données

- Toutes les requêtes SQL sont dans db/readers.py ou db/writers.py.
  Jamais de SQL inline dans pipeline/ ou models/.
- Utilise des requêtes paramétrées. Jamais de f-string pour construire du SQL.
- Chaque fonction DB documente la table qu'elle touche et le type d'opération
  (lecture / écriture).
- Les insertions en masse utilisent execute_values (psycopg2) pour la performance.

## Logging

- Utilise le module logging standard, configuré dans main.py.
- Niveau INFO pour les événements nominaux (forecast généré, modèle chargé).
- Niveau WARNING pour les situations dégradées récupérables (retry API, MAPE élevé).
- Niveau ERROR pour les échecs qui nécessitent une intervention.
- Inclure systématiquement site_id dans chaque log pour faciliter le débogage
  multi-sites.
- Pas de print(). Jamais.

## Tests

- Chaque nouvelle fonction publique dans pipeline/, models/ et db/ doit avoir
  au moins un test unitaire dans tests/ avec le même chemin miroir
  (pipeline/forecast.py → tests/pipeline/test_forecast.py).
- Les tests n'appellent jamais la base de données ni les APIs externes.
  Utilise des fixtures et des mocks (unittest.mock ou pytest-mock).
- Nomme les tests de façon descriptive :
  `test_compute_mape_returns_zero_when_perfect_forecast`
  `test_run_forecast_skips_site_on_meteo_error`
- Un test = un comportement vérifié. Pas de tests qui vérifient 5 choses à la fois.
- Après chaque nouvelle feature, lance pytest et assure-toi que tous les tests
  passent avant de continuer.

## Sécurité

- Aucune credential (clé API, mot de passe DB, clé VPN) dans le code ou dans
  un fichier versionné. Tout passe par des variables d'environnement.
- Le fichier .env est dans .gitignore. Un .env.example avec des valeurs fictives
  est versionné à la place.
- Pas de log de valeurs sensibles (tokens, credentials).

## Docker

- Le Dockerfile utilise python:3.11-slim comme base.
- Les secrets sont injectés via des variables d'environnement, jamais copiés
  dans l'image.

## Documentation

- Mets à jour le README.md à chaque nouvelle feature ou changement de
  comportement. Le README doit toujours refléter l'état réel du code.
- Le README contient : description du service, variables d'environnement
  requises, commandes pour lancer les tests, commandes pour lancer le service
  en local.
- Chaque job du scheduler est documenté dans le README avec son heure de
  déclenchement et ce qu'il fait.
- Si tu ajoutes un point ouvert ou une décision d'architecture non tranchée,
  note-le dans une section ## Points ouverts du README.

## Workflow

- Avant de commencer une nouvelle feature, résume en une phrase ce que tu vas
  faire et pourquoi.
- Après chaque feature complète : tests passent, README à jour, pas de code
  commenté traînant, pas de TODO sans ticket associé.
- Si tu identifies une ambiguïté dans les specs (CLAUDE.md ou README), pose
  la question avant de coder plutôt que de faire un choix silencieux.
