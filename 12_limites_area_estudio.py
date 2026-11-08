# -*- coding: utf-8 -*-
"""
12_limites_area_estudio.py
Describe el polígono del área de estudio (Amazonía colombiana) para la Sección 2.1:
  - origen probable de la capa (CRS, columnas, atributos)
  - límites geográficos (extremos N, S, E, O) en grados decimales y en grados-minutos
  - punto exacto donde está cada extremo (para saber qué lugar marca el límite)
  - área geodésica en km² (sobre el elipsoide WGS84, sin depender de la proyección)
  - número aproximado de píxeles ABI de ≈2 km
Sale por consola y en limites_area_estudio.txt (junto al script).

Requisitos: geopandas, shapely, pyproj (vienen con geopandas).
"""
from pathlib import Path
import geopandas as gpd
from pyproj import Geod
from shapely.ops import unary_union

# ======================== CONFIGURA AQUÍ ========================
RUTA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Revisión 1\Poligono Amazonia.shp"  # .shp, .gpkg o .geojson
CAPA = None          # solo para .gpkg con varias capas, p. ej. "regiones"
COLUMNA = None       # columna con el nombre de la región, p. ej. "REGION"; None = intentar detectarla
VALOR = "amaz"       # texto que identifica la Amazonía en esa columna (sin distinguir mayúsculas)
USAR_TODA_LA_CAPA = True   # True: une todos los polígonos (capa ya recortada al área de estudio, p. ej. por departamento)
# ================================================================

salida = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s); salida.append(s)

def dms(v, eje):
    hemi = ("N" if v >= 0 else "S") if eje == "lat" else ("E" if v >= 0 else "W")
    v = abs(v); g = int(v); m = (v - g) * 60
    tot = round(v * 60); gr, mr = divmod(tot, 60)          # redondeo al minuto sin producir 60′
    return f"{g}°{m:05.2f}′ {hemi}  (≈ {gr}°{mr:02d}′ {hemi})"

gdf = gpd.read_file(RUTA, layer=CAPA) if CAPA else gpd.read_file(RUTA)
out("Archivo:", RUTA)
out("CRS original:", gdf.crs)
out("Número de polígonos:", len(gdf))
out("Columnas:", list(gdf.columns))
out("\nPrimeras filas de atributos (sirven para identificar la fuente: IGAC, SINCHI, etc.):")
out(gdf.drop(columns=gdf.geometry.name).head(10).to_string())

# --- seleccionar la Amazonía si la capa trae varias regiones
if len(gdf) > 1 and not USAR_TODA_LA_CAPA:
    cols = [COLUMNA] if COLUMNA else [c for c in gdf.columns if c != gdf.geometry.name]
    sel = None
    for c in cols:
        m = gdf[c].astype(str).str.contains(VALOR, case=False, na=False)
        if m.any():
            sel = gdf[m]; out(f"\nSe seleccionaron {m.sum()} polígono(s) donde '{c}' contiene '{VALOR}':",
                              sorted(gdf.loc[m, c].astype(str).unique()))
            break
    if sel is None:
        out(f"\nNo se encontró '{VALOR}' en ninguna columna de texto: se usa toda la capa. Ajusta COLUMNA y VALOR.")
        sel = gdf
else:
    sel = gdf

if sel.crs is None:
    out("\nAVISO: la capa no tiene CRS definido; se asume WGS84 (EPSG:4326). Verifica que las coordenadas estén en grados.")
    sel = sel.set_crs(4326)
geo = sel.to_crs(4326)
area_union = unary_union(geo.geometry.values)

# --- área de cada polígono: si coincide con el departamento completo, la capa NO está recortada a la Amazonía
geod = Geod(ellps="WGS84")
nom = "NAME_1" if "NAME_1" in geo.columns else None
out("\nÁREA POR POLÍGONO (km²) Y SUS LÍMITES")
for i, r in geo.iterrows():
    a = abs(geod.geometry_area_perimeter(r.geometry)[0]) / 1e6
    b = r.geometry.bounds
    out(f"  {str(r[nom]) if nom else i:<12} {a:>10,.0f}   lat {b[1]:7.3f} a {b[3]:6.3f} | lon {b[0]:8.3f} a {b[2]:8.3f}".replace(",", " "))

# --- extremos
minx, miny, maxx, maxy = area_union.bounds
out("\nLÍMITES GEOGRÁFICOS (WGS84)")
out(f"  Norte : {maxy:.4f}  ->  {dms(maxy, 'lat')}")
out(f"  Sur   : {miny:.4f}  ->  {dms(miny, 'lat')}")
out(f"  Oeste : {minx:.4f}  ->  {dms(minx, 'lon')}")
out(f"  Este  : {maxx:.4f}  ->  {dms(maxx, 'lon')}")

# --- dónde está cada extremo (punto del contorno que lo produce)
pts = []
for g in getattr(area_union, "geoms", [area_union]):
    pts.extend(g.exterior.coords)
for nombre, f in [("Norte", lambda p: p[1]), ("Sur", lambda p: -p[1]), ("Oeste", lambda p: -p[0]), ("Este", lambda p: p[0])]:
    p = max(pts, key=f)
    out(f"  Punto extremo {nombre}: lat {p[1]:.4f}, lon {p[0]:.4f}  (búscalo en Google Maps para ver qué lugar es)")

# --- área geodésica total
area_km2 = abs(geod.geometry_area_perimeter(area_union)[0]) / 1e6
out(f"\nÁREA GEODÉSICA: {area_km2:,.0f} km²".replace(",", " "))
out(f"  Manuscrito: 483 163 km²  |  diferencia: {area_km2 - 483163:,.0f} km²".replace(",", " "))
out(f"  Píxeles ABI aproximados (≈4 km² cada uno): {area_km2 / 4:,.0f}".replace(",", " "))

c = area_union.centroid
out(f"\nCentroide: lat {c.y:.4f}, lon {c.x:.4f}")

f = Path(__file__).with_name("limites_area_estudio.txt")
f.write_text("\n".join(salida), encoding="utf-8")
print("\nGuardado:", f)