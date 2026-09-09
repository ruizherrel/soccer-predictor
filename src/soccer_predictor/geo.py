"""Team home-city geography (coordinates, altitude) for altitude/travel
features.

Only populated for Mexico: altitude is a well-documented real effect there
(Mexico City ~2240m vs sea-level opponents like Mazatlán or Tijuana is a
~2200m swing), while every European league in this project sits within a
few hundred meters of sea level, so there's no comparable signal to justify
building out a full per-team lookup for them yet. Coordinates/altitudes are
approximate city-level figures (not exact stadium GPS) — accurate enough to
capture the real effect, which is on the order of kilometers/kilometers of
altitude, not meters.
"""
from __future__ import annotations

import math

# team -> (latitude, longitude, altitude_m)
TEAM_LOCATIONS: dict[str, tuple[float, float, float]] = {
    "América": (19.43, -99.13, 2240),
    "Atlante": (19.43, -99.13, 2240),
    "Cruz Azul": (19.43, -99.13, 2240),
    "Pumas UNAM": (19.43, -99.13, 2240),
    "Atlas": (20.67, -103.35, 1570),
    "CD Guadalajara": (20.67, -103.35, 1570),
    "Atletico de San Luis": (22.15, -100.98, 1877),
    "Juárez": (31.69, -106.42, 1137),
    "León": (21.12, -101.68, 1800),
    "Mazatlán": (23.25, -106.41, 10),
    "Monterrey": (25.67, -100.31, 540),
    "Tigres UANL": (25.67, -100.31, 540),
    "Necaxa": (21.88, -102.29, 1880),
    "Pachuca": (20.12, -98.73, 2316),
    "Puebla": (19.04, -98.20, 2135),
    "Querétaro": (20.59, -100.39, 1820),
    "Santos Laguna": (25.54, -103.41, 1120),
    "Tijuana": (32.52, -117.02, 20),
    "Toluca": (19.28, -99.65, 2680),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def altitude_features(home_team: str, away_team: str) -> dict[str, float]:
    """NaN whenever either team's location is unknown (every non-Mexico
    team right now) — XGBoost already handles missing features natively
    elsewhere in this project (e.g. the first season's Poisson features)."""
    home_loc = TEAM_LOCATIONS.get(home_team)
    away_loc = TEAM_LOCATIONS.get(away_team)
    if home_loc is None or away_loc is None:
        return {
            "home_altitude_m": float("nan"),
            "altitude_delta_m": float("nan"),
            "away_travel_km": float("nan"),
        }

    home_lat, home_lon, home_alt = home_loc
    away_lat, away_lon, away_alt = away_loc
    return {
        "home_altitude_m": float(home_alt),
        "altitude_delta_m": float(home_alt - away_alt),
        "away_travel_km": haversine_km(home_lat, home_lon, away_lat, away_lon),
    }
