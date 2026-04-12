"""
Générateur de données de consommation synthétiques pour les tests.

Produit un an de mesures au pas 15 min représentatives d'un site industriel
2×8h avec un pic de ~700 kW.

Usage :
    uv run python tests/fixtures/generate_load_data.py

Sortie : tests/fixtures/load_history_2025.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

SEED = 42
OUTPUT_FILE = Path(__file__).parent / "load_history_2025.csv"

# Paramètres du profil de charge (kW)
BASE_KW = 100.0          # charge de base permanente
EQUIPE_1_KW = 400.0      # surcroît équipe 1 (7h–15h, lun–ven)
EQUIPE_2_KW = 330.0      # surcroît équipe 2 (15h–23h, lun–ven)
EQUIPE_SAMEDI_KW = 150.0 # surcroît samedi partiel (8h–16h)
CLIM_KW = 80.0           # surcroît climatisation quand temp > 25°C
RAMP_STEPS = 2           # nombre de pas 15 min pour la montée/descente machine

BRUIT_CONSO_PCT = 0.05   # écart-type du bruit gaussien sur la conso (5 %)
BRUIT_TEMP_C = 1.0       # écart-type du bruit gaussien sur la température

# Jours fériés France 2025
JOURS_FERIES_2025 = set(holidays.France(years=2025).keys())


def temperature_sinusoide(timestamp: pd.Timestamp) -> float:
    """
    Température saisonnière synthétique.

    Sinusoïde : min ~4°C en janvier, max ~29°C en juillet.
    """
    # Jour de l'année normalisé entre 0 et 2π, décalé pour que le max soit en juillet (jour ~196)
    day_of_year = timestamp.day_of_year
    # pic en été : cos(0) = 1 → max en jour 196 (~15 juillet)
    phase = 2 * math.pi * (day_of_year - 196) / 365
    amplitude = (29.0 - 4.0) / 2   # 12.5°C
    mean_temp = (29.0 + 4.0) / 2   # 16.5°C
    return mean_temp + amplitude * math.cos(phase)


def puissance_cible(timestamp: pd.Timestamp, temperature_c: float) -> float:
    """
    Puissance de consommation cible (avant bruit) pour un pas 15 min.

    Règles :
    - Dimanche ou jour férié → base uniquement
    - Samedi 8h–16h         → base + équipe partielle
    - Lun–Ven 7h–15h        → base + équipe 1
    - Lun–Ven 15h–23h       → base + équipe 2
    - Sinon                  → base
    - Si temp > 25°C         → +CLIM_KW
    """
    is_ferie = timestamp.date() in JOURS_FERIES_2025
    weekday = timestamp.weekday()  # 0=lundi, 6=dimanche
    hour = timestamp.hour + timestamp.minute / 60.0

    puissance = BASE_KW

    if is_ferie or weekday == 6:
        pass  # base seulement
    elif weekday == 5:  # samedi
        if 8.0 <= hour < 16.0:
            puissance += EQUIPE_SAMEDI_KW
    else:  # lundi–vendredi
        if 7.0 <= hour < 15.0:
            puissance += EQUIPE_1_KW
        elif 15.0 <= hour < 23.0:
            puissance += EQUIPE_2_KW

    if temperature_c > 25.0:
        puissance += CLIM_KW

    return puissance


def appliquer_ramp(series: pd.Series) -> pd.Series:
    """
    Lisse les transitions de puissance sur RAMP_STEPS pas (montée/descente machine).

    On applique une moyenne glissante uniquement aux pas où la puissance change
    fortement (variation > 50 kW).
    """
    result = series.copy().astype(float)
    valeurs = result.values.copy()

    for i in range(1, len(valeurs) - 1):
        delta = abs(valeurs[i] - valeurs[i - 1])
        if delta > 50:
            # Interpolation linéaire entre les deux niveaux
            debut = valeurs[i - 1]
            fin = valeurs[i]
            for k in range(RAMP_STEPS):
                if i + k < len(valeurs):
                    valeurs[i + k] = debut + (fin - debut) * (k + 1) / (RAMP_STEPS + 1)

    result[:] = valeurs
    return result


def generer_donnees() -> pd.DataFrame:
    """Génère le DataFrame complet d'un an de mesures synthétiques."""
    rng = np.random.default_rng(SEED)

    # Index : 2025-01-01 00:00 → 2025-12-31 23:45 (Europe/Paris)
    tz = "Europe/Paris"
    index = pd.date_range(
        start="2025-01-01 00:00:00",
        end="2025-12-31 23:45:00",
        freq="15min",
        tz=tz,
    )

    # Température synthétique
    temp_base = np.array([temperature_sinusoide(ts) for ts in index])
    bruit_temp = rng.normal(0, BRUIT_TEMP_C, size=len(index))
    temperature_c = np.round(temp_base + bruit_temp, 2)

    # Puissance cible (avant bruit)
    conso_base = np.array(
        [puissance_cible(ts, temperature_c[i]) for i, ts in enumerate(index)]
    )

    # Lissage des transitions de shifts
    conso_lissee = appliquer_ramp(pd.Series(conso_base))

    # Bruit gaussien 5 %
    bruit_conso = rng.normal(0, BRUIT_CONSO_PCT, size=len(index))
    conso_kw = np.maximum(conso_lissee.values * (1 + bruit_conso), 0)
    conso_kw = np.round(conso_kw, 2)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "conso_kw": conso_kw,
            "temperature_c": temperature_c,
        }
    )

    return df


def main() -> None:
    print("Génération des données synthétiques...")
    df = generer_donnees()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  {len(df):,} lignes écrites → {OUTPUT_FILE}")
    print(f"  Période   : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"  conso_kw  : min={df['conso_kw'].min():.1f}  max={df['conso_kw'].max():.1f}  "
          f"mean={df['conso_kw'].mean():.1f}")
    print(f"  temp (°C) : min={df['temperature_c'].min():.1f}  "
          f"max={df['temperature_c'].max():.1f}  mean={df['temperature_c'].mean():.1f}")


if __name__ == "__main__":
    main()
