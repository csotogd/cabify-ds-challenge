"""Estimación del impacto económico de la desviación entre ruta estimada y real.

El precio de Cabify se fija ANTES del viaje con la ruta estimada. Si el recorrido real
difiere, la diferencia entre lo cobrado y lo que ese recorrido habría costado es el
error de tarificación. La tarifa base y la mínima son idénticas en ambos cálculos y se
cancelan al restar, así que solo hacen falta los precios por kilómetro y por minuto.

    Δ€ = €/km · (km_real − km_estimado) + €/min · (min_real − min_estimado)

Δ€ > 0: el trayecto real valía más de lo cobrado.
Δ€ < 0: el cliente pagó por encima de lo que costaba su trayecto real.

LIMITACIONES, que son grandes y deliberadamente explícitas:

1. Las tarifas son públicas y actuales (octubre de 2026); el archivo es de diciembre de
   2021. No reconstruyen lo que se cobró entonces. Sirven para dimensionar la exposición
   con precios de hoy, no para auditar cobros pasados. En Argentina, con la inflación del
   periodo, la cifra histórica sería incomparable.
2. Se usa la categoría estándar. No conocemos la categoría real de cada viaje.
3. Se ignoran peajes, suplementos, tarifa dinámica, esperas y promociones.
4. Las duraciones salen de OSRM con un perfil fijo de coche, sin tráfico histórico.
5. Colombia y Ecuador usan estimaciones; la fuente oficial no publica un valor único.

Es una aproximación de orden de magnitud para priorizar, no una cifra contable.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .data import ROOT, load_data
from .features import build_features

CONSULTED = '2026-10-19'

# Margen EBITDA del grupo Cabify en 2025: 27,2 M€ sobre 833,3 M€ de ingresos.
# Fuente: resultados publicados en prensa económica (Forbes España, El Economista), junio de 2026.
# Es un margen de grupo, no el margen unitario de un viaje concreto: lo usamos solo para
# dimensionar la desviación frente a la rentabilidad típica, no como contabilidad por trayecto.
EBITDA_MARGIN = 27.2/833.3

# Tarifa estándar de Cabify por ciudad, en moneda local, y tipo de cambio a euros.
# Fuentes: páginas oficiales cabify.com/<pais>/tarifas/<ciudad>, consultadas en CONSULTED.
@dataclass(frozen=True)
class Tariff:
    currency: str
    per_km_local: float
    per_min_local: float
    eur_per_unit: float       # unidades de moneda local por 1 EUR
    source: str
    base_local: float = 0.    # tarifa base: se cancela en Δ€, pero cuenta en el precio del viaje
    min_local: float = 0.     # tarifa mínima aplicable
    estimated: bool = False   # True si no procede de una cifra única publicada

    @property
    def per_km_eur(self): return self.per_km_local/self.eur_per_unit
    @property
    def per_min_eur(self): return self.per_min_local/self.eur_per_unit
    @property
    def base_eur(self): return self.base_local/self.eur_per_unit
    @property
    def min_eur(self): return self.min_local/self.eur_per_unit

SPAIN = Tariff('EUR', .76, .12, 1., 'cabify.com/es/tarifas/madrid · categoría Cabify', 1.25, 4.33)
ARGENTINA = Tariff('ARS', 434., 81., 1705.87, 'cabify.com/ar/tarifas/buenos-aires · categoría Cabify', 1325., 3569.)
CHILE = Tariff('CLP', 440., 85., 1103.49, 'cabify.com/cl/tarifas/santiago · categoría Cabify', 550., 1450.)
PERU = Tariff('PEN', 1.14, .21, 3.85635, 'cabify.com/pe/tarifas/lima · categoría Cabify, zona 1', 4.48, 7.00)
MEXICO = Tariff('MXN', 8.50, 3.25, 20.6871, 'Prensa y agregadores; Cabify no publica tabla única', 25., 45., estimated=True)
URUGUAY = Tariff('UYU', 19.80, 6.40, 45.03, 'cabify.com/uy/montevideo · categoría Cabify', 86., 190.)
# Colombia publica rangos dependientes de demanda (Medellín: 245–1.403 $/km, 51–495 $/min).
# Tomamos la media geométrica del rango, que cae entre Argentina y Perú en euros por km.
COLOMBIA = Tariff('COP', float(np.sqrt(245*1403)), float(np.sqrt(51*495)), 3737.38,
                  'cabify.com/co/tarifas/medellin · media geométrica del rango publicado',
                  float(np.sqrt(1023*8250)), float(np.sqrt(2000*9075)), estimated=True)
# Ecuador está dolarizado y no publica tabla accesible: usamos la mediana latinoamericana.
_LATAM = [ARGENTINA, CHILE, PERU, MEXICO, URUGUAY, COLOMBIA]
ECUADOR = Tariff('USD', float(np.median([t.per_km_eur for t in _LATAM])),
                 float(np.median([t.per_min_eur for t in _LATAM])), 1.,
                 'Sin tabla publicada accesible; mediana de las demás ciudades del archivo',
                 float(np.median([t.base_eur for t in _LATAM])),
                 float(np.median([t.min_eur for t in _LATAM])), estimated=True)

CITY_TARIFF = {
    'Madrid': SPAIN, 'Barcelona': SPAIN, 'Valencia': SPAIN, 'Sevilla': SPAIN,
    'Buenos Aires': ARGENTINA, 'Córdoba': ARGENTINA, 'Mendoza': ARGENTINA,
    'Bogotá': COLOMBIA, 'Medellín': COLOMBIA, 'Cali': COLOMBIA,
    'Santiago': CHILE, 'Lima': PERU, 'Quito': ECUADOR,
    'Ciudad de México': MEXICO, 'Montevideo': URUGUAY,
}
FALLBACK = ECUADOR   # 'Otras zonas': misma mediana latinoamericana

def tariff_of(region): return CITY_TARIFF.get(region, FALLBACK)

def build_impact():
    """Un DataFrame por viaje con la desviación de distancia, duración y euros."""
    df, _ = load_data()
    geom = build_features().loc[df.index]
    graph = pd.read_json(ROOT/'data/cache/roads/graph_features.json', orient='index').loc[df.index]

    km_est, km_real = geom.length_est/1000, geom.length_real/1000
    reliable = graph.both_reliable.astype(bool)
    # Velocidad implícita de OSRM, por ciudad, para los viajes sin encaje fiable.
    speed = (graph.est_partial_length_m/1000)/(graph.est_partial_duration_s/3600)
    city_speed = speed[reliable].groupby(df.region).median()
    overall_speed = float(speed[reliable].median())
    fallback_speed = df.region.map(city_speed).fillna(overall_speed)

    min_est = np.where(reliable, graph.est_partial_duration_s/60, 60*km_est/fallback_speed)
    min_real = np.where(reliable, graph.real_partial_duration_s/60, 60*km_real/fallback_speed)

    per_km = df.region.map(lambda r: tariff_of(r).per_km_eur)
    per_min = df.region.map(lambda r: tariff_of(r).per_min_eur)
    delta_eur = per_km*(km_real-km_est) + per_min*(min_real-min_est)
    # Precio de referencia del viaje tal y como se habría cobrado con la estimación.
    # Precio efectivamente cobrado: base + parte variable, nunca por debajo de la mínima.
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
        'caveat': 'Tarifas públicas de 2026 sobre un archivo de 2021; categoría estándar; '
                  'sin peajes, dinámica ni promociones; duraciones de OSRM sin tráfico.',
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
