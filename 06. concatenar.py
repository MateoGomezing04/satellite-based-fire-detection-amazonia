# -*- coding: utf-8 -*-
"""
excel_a_csv_concatenar.py -- Convierte cada Excel mensual recortado a CSV y los une en un solo CSV.

Para cada .xlsx de la carpeta (se omiten los 'consolidado*' para no duplicar filas):
  - lee la hoja 'datos' y sus continuaciones ('datos_2', ...)
  - guarda un CSV por mes en la subcarpeta 'csv' (mismo nombre, extensión .csv)
Luego une todos en 'csv\\dataset_amazonia_2024.csv', quita filas píxel-imagen repetidas e imprime un resumen.

Formato del CSV: separador coma, punto decimal, UTF-8 (lo que esperan pandas, R, scikit-learn).
Si lo va a abrir en Excel en español y ve todo en una columna, use SEPARADOR = ";" y DECIMAL = ",".

USO (PowerShell):  python excel_a_csv_concatenar.py
                   python excel_a_csv_concatenar.py --carpeta "C:\\otra\\carpeta"
Requiere: pip install pandas openpyxl
"""
import os, sys, glob, argparse
import pandas as pd

CARPETA = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\2024\salida_goes_areas"
NOMBRE_FINAL = "dataset_amazonia_2024 final.csv"
SEPARADOR, DECIMAL = ",", "."

def leer_datos(ruta):
    hojas = pd.read_excel(ruta, sheet_name=None)
    partes = [h for n, h in hojas.items() if (n == "datos" or n.startswith("datos_")) and "label" in h.columns]
    return pd.concat(partes, ignore_index=True) if partes else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--carpeta", default=CARPETA); args = ap.parse_args()
    if not os.path.isdir(args.carpeta): sys.exit(f"[DETENIDO] No existe la carpeta: {args.carpeta}")
    out_dir = os.path.join(args.carpeta, "csv"); os.makedirs(out_dir, exist_ok=True)
    archivos = sorted(f for f in glob.glob(os.path.join(args.carpeta, "*.xlsx"))
                      if not os.path.basename(f).startswith("~$") and not os.path.basename(f).lower().startswith("consolidado"))
    if not archivos: sys.exit(f"[DETENIDO] No hay Excel en {args.carpeta}")
    print(f"Excel encontrados: {len(archivos)}")
    todos = []
    for f in archivos:
        try:
            df = leer_datos(f)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(f)}: {type(e).__name__}: {e}"); continue
        if df is None or df.empty:
            print(f"  {os.path.basename(f)}: sin filas de datos -> se omite"); continue
        df.insert(0, "archivo_origen", os.path.basename(f))
        df.to_csv(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".csv"), index=False, sep=SEPARADOR, decimal=DECIMAL, encoding="utf-8")
        print(f"  {os.path.basename(f)}: {len(df):,} filas -> CSV")
        todos.append(df)
    if not todos: sys.exit("No se encontraron datos para unir.")
    df = pd.concat(todos, ignore_index=True)
    if {"image_id", "pixel_id"}.issubset(df.columns):
        dup = df.duplicated(["image_id", "pixel_id"])
        if dup.any(): print(f"[AVISO] {int(dup.sum()):,} filas píxel-imagen repetidas; se conserva la primera"); df = df[~dup]
    final = os.path.join(out_dir, NOMBRE_FINAL)
    df.to_csv(final, index=False, sep=SEPARADOR, decimal=DECIMAL, encoding="utf-8")
    print(f"\nCSV unido: {final}\nFilas: {len(df):,} | columnas: {df.shape[1]}")
    if "mes" in df.columns and "label" in df.columns:
        print("\nFilas y positivos por mes:")
        print(df.groupby("mes").label.agg(filas="size", positivos="sum").to_string())
    if "neg_type" in df.columns: print("\nPor clase:"); print(df.neg_type.value_counts().to_string())

if __name__ == "__main__":
    main()