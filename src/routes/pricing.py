"""Estimating the economic impact of the gap between the estimated and the real route.

Cabify sets the fare BEFORE the trip using the estimated route. If the actual trip
differs, the gap between what was charged and what that trip would have cost is the
pricing error. The base fare and the minimum fare are identical in both calculations and
cancel out when subtracting, so only the per-kilometre and per-minute prices are needed.

    delta_eur = eur_per_km * (km_real - km_est) + eur_per_min * (min_real - min_est)

delta_eur > 0: the real trip was worth more than what was charged.
delta_eur < 0: the customer paid more than their real trip cost.

LIMITATIONS, which are large and deliberately explicit:

1. The fares are public and current (October 2026); the file is from December
   2021. They do not reconstruct what was charged then. They serve to size the exposure
   at today's prices, not to audit past charges. In Argentina, with the inflation of the
   period, the historical figure would not be comparable.
2. The standard category is used. We do not know the real category of each trip.
3. Tolls, surcharges, surge pricing, waiting time and promotions are ignored.
4. Durations come from OSRM with a fixed car profile, with no historical traffic.
5. Colombia and Ecuador use estimates; the official source publishes no single value.

This is an order-of-magnitude approximation for prioritising, not an accounting figure.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .data import ROOT, load_data
from .features import build_features

CONSULTED = '2026-10-19'

# Cabify group EBITDA margin in 2025: EUR 27.2M on EUR 833.3M of revenue.
# Source: results published in the financial press (Forbes Espana, El Economista), June 2026.
# This is a group margin, not the unit margin of a particular trip: we use it only to
# size the deviation against typical profitability, not as per-trip accounting.
EBITDA_MARGIN = 27.2/833.3

# Cabify standard fare per city, in local currency, plus the exchange rate to euros.
# Sources: official pages cabify.com/<country>/tarifas/<city>, consulted on CONSULTED.
@dataclass(frozen=True)
class Tariff:
    currency: str
    per_km_local: float
    per_min_local: float
    eur_per_unit: float       # units of local currency per 1 EUR
    source: str
    base_local: float = 0.    # base fare: cancels in delta_eur, but counts in the trip price
    min_local: float = 0.     # applicable minimum fare
    estimated: bool = False   # True when not taken from a single published figure

    @property
    def per_km_eur(self): return self.per_km_local/self.eur_per_unit
    @property
    def per_min_eur(self): return self.per_min_local/self.eur_per_unit
    @property
    def base_eur(self): return self.base_local/self.eur_per_unit
    @property
    def min_eur(self): return self.min_local/self.eur_per_unit

SPAIN = Tariff('EUR', .76, .12, 1., 'cabify.com/es/tarifas/madrid · Cabify category', 1.25, 4.33)
ARGENTINA = Tariff('ARS', 434., 81., 1705.87, 'cabify.com/ar/tarifas/buenos-aires · Cabify category', 1325., 3569.)
CHILE = Tariff('CLP', 440., 85., 1103.49, 'cabify.com/cl/tarifas/santiago · Cabify category', 550., 1450.)
PERU = Tariff('PEN', 1.14, .21, 3.85635, 'cabify.com/pe/tarifas/lima · Cabify category, zone 1', 4.48, 7.00)
MEXICO = Tariff('MXN', 8.50, 3.25, 20.6871, 'Press and aggregators; Cabify publishes no single table', 25., 45., estimated=True)
URUGUAY = Tariff('UYU', 19.80, 6.40, 45.03, 'cabify.com/uy/montevideo · Cabify category', 86., 190.)
# Colombia publishes demand-dependent ranges (Medellin: 245-1,403 $/km, 51-495 $/min).
# We take the geometric mean of the range, which falls between Argentina and Peru in euros per km.
COLOMBIA = Tariff('COP', float(np.sqrt(245*1403)), float(np.sqrt(51*495)), 3737.38,
                  'cabify.com/co/tarifas/medellin · geometric mean of the published range',
                  float(np.sqrt(1023*8250)), float(np.sqrt(2000*9075)), estimated=True)
# Ecuador is dollarised and publishes no accessible table: we use the Latin American median.
_LATAM = [ARGENTINA, CHILE, PERU, MEXICO, URUGUAY, COLOMBIA]
ECUADOR = Tariff('USD', float(np.median([t.per_km_eur for t in _LATAM])),
                 float(np.median([t.per_min_eur for t in _LATAM])), 1.,
                 'No accessible published table; median of the other cities in the file',
                 float(np.median([t.base_eur for t in _LATAM])),
                 float(np.median([t.min_eur for t in _LATAM])), estimated=True)

CITY_TARIFF = {
    'Madrid': SPAIN, 'Barcelona': SPAIN, 'Valencia': SPAIN, 'Sevilla': SPAIN,
    'Buenos Aires': ARGENTINA, 'Córdoba': ARGENTINA, 'Mendoza': ARGENTINA,
    'Bogotá': COLOMBIA, 'Medellín': COLOMBIA, 'Cali': COLOMBIA,
    'Santiago': CHILE, 'Lima': PERU, 'Quito': ECUADOR,
    'Ciudad de México': MEXICO, 'Montevideo': URUGUAY,
}
FALLBACK = ECUADOR   # 'Otras zonas': same Latin American median

def tariff_of(region): return CITY_TARIFF.get(region, FALLBACK)

def build_impact():
    """One DataFrame row per journey with the distance, duration and euro deviation."""
    df, _ = load_data()
    geom = build_features().loc[df.index]
    graph = pd.read_json(ROOT/'data/cache/roads/graph_features.json', orient='index').loc[df.index]

    km_est, km_real = geom.length_est/1000, geom.length_real/1000
    reliable = graph.both_reliable.astype(bool)
    # OSRM implied speed, per city, for journeys without a reliable map match.
    speed = (graph.est_partial_length_m/1000)/(graph.est_partial_duration_s/3600)
    city_speed = speed[reliable].groupby(df.region).median()
    overall_speed = float(speed[reliable].median())
    fallback_speed = df.region.map(city_speed).fillna(overall_speed)

    min_est = np.where(reliable, graph.est_partial_duration_s/60, 60*km_est/fallback_speed)
    min_real = np.where(reliable, graph.real_partial_duration_s/60, 60*km_real/fallback_speed)

    per_km = df.region.map(lambda r: tariff_of(r).per_km_eur)
    per_min = df.region.map(lambda r: tariff_of(r).per_min_eur)
    delta_eur = per_km*(km_real-km_est) + per_min*(min_real-min_est)
    # Reference trip price as it would have been charged using the estimate.
    # Price actually charged: base + variable part, never below the minimum fare.
    base = df.region.map(lambda r: tariff_of(r).base_eur)
    minimum = df.region.map(lambda r: tariff_of(r).min_eur)
    charged = np.maximum(base + per_km*km_est + per_min*min_est, minimum)

    return pd.DataFrame({
        'region': df.region, 'split': df.split, 'status': df.status,
        'n_votes': df.n_votes, 'target_soft': df.target_soft, 'target_hard': df.target_hard,
        'km_est': km_est, 'km_real': km_real, 'delta_km': km_real-km_est,
        'min_est': min_est, 'min_real': min_real, 'delta_min': min_real-min_est,
        'duration_from_osrm': reliable, 'per_km_eur': per_km, 'per_min_eur': per_min,
        'charged_eur': charged, 'delta_eur': delta_eur,
        'abs_delta_eur': delta_eur.abs(), 'margin_eur': EBITDA_MARGIN*charged,
        'deviation_over_margin': delta_eur.abs()/(EBITDA_MARGIN*charged).clip(lower=1e-4),
        'relative_delta': delta_eur/charged.clip(lower=.01)}, index=df.index)

def tariff_table():
    seen, rows = {}, []
    for city, t in sorted(CITY_TARIFF.items()):
        seen.setdefault(id(t), (t, []))[1].append(city)
    for t, cities in seen.values():
        rows.append({'Ciudades': ', '.join(cities), **asdict(t),
                     'per_km_eur': t.per_km_eur, 'per_min_eur': t.per_min_eur})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    impact = build_impact()
    out = ROOT/'results/routes_euros'; out.mkdir(parents=True, exist_ok=True)
    summary = {'consulted': CONSULTED,
        'caveat': 'Public 2026 fares applied to a 2021 file; standard category; '
                  'no tolls, surge pricing or promotions; OSRM durations with no traffic.',
        'duration_from_osrm': int(impact.duration_from_osrm.sum()), 'n': len(impact),
        'tariffs': json.loads(tariff_table().to_json(orient='records')),
        'abs_delta_eur': {k: float(v) for k, v in
                          impact.abs_delta_eur.describe(percentiles=[.5, .75, .95, .99]).items()},
        'by_label': {str(int(k)): {'n': int(len(g)), 'median_abs_eur': float(g.abs_delta_eur.median()),
                                   'p95_abs_eur': float(g.abs_delta_eur.quantile(.95)),
                                   'mean_signed_eur': float(g.delta_eur.mean())}
                     for k, g in impact[impact.target_hard.notna()].groupby('target_hard')}}
    (out/'impact_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    impact.to_json(out/'impact.json', orient='index', indent=1)
    print(json.dumps(summary['by_label'], indent=2, ensure_ascii=False))
    print('abs |Δ€| mediana', round(impact.abs_delta_eur.median(), 3),
          '· P95', round(impact.abs_delta_eur.quantile(.95), 2))
