#!/usr/bin/env python3
"""
Script para analizar automÃ¡ticamente los resultados del MÃ©todo ASR
Genera tablas comparativas y calcula el âˆ†F1â€‘RS para cada modelo
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json


def extract_test_metrics(folder_path):
    """Extrae mÃ©tricas de test de una carpeta de experimento"""
    metrics_file = Path(folder_path) / "test_metrics.csv"

    if not metrics_file.exists():
        return None

    df = pd.read_csv(metrics_file)
    return df.iloc[0].to_dict()


def find_experiment_folders(base_dir="."):
    """Encuentra todas las carpetas de experimentos"""
    folders = []
    for item in Path(base_dir).iterdir():
        if item.is_dir() and item.name.startswith("main_"):
            folders.append(item)
    return sorted(folders)


def categorize_experiment(folder_name):
    """Categoriza el experimento segÃºn su nombre"""
    name = folder_name.name

    # Determinar fase
    if "_fase1_" in name:
        fase = "Fase 1: Multi-etiqueta"
    elif "_fase3_" in name:
        fase = "Fase 3: Contraste Binario"
    else:
        fase = "Otro"

    # Determinar dataset
    if "_edc_" in name:
        dataset = "EDC"
    elif "_odir5k_" in name:
        dataset = "ODIR-5K"
    else:
        dataset = "Desconocido"

    # Determinar modelo
    if "efficientnet_b0" in name:
        modelo = "EfficientNet-B0"
    elif "convnext_base" in name:
        modelo = "ConvNeXt-Base"
    elif "swin_tiny" in name:
        modelo = "Swin-Tiny"
    elif "swin_base" in name:
        modelo = "Swin-Base"
    elif "swin_small" in name:
        modelo = "Swin-Small"
    else:
        modelo = "Desconocido"

    # Determinar tipo de F1â€‘RS (solo para Fase 3)
    f1rs_type = None
    contraste = None
    if "_F1RS_max_" in name:
        f1rs_type = "F1â€‘RS MÃ¡ximo"
        if "D_vs_notD" in name:
            contraste = "D vs Â¬D"
        elif "N_vs_All" in name:
            contraste = "N vs All"
    elif "_F1RS_ctrl_" in name:
        f1rs_type = "F1â€‘RS Control"
        if "G_vs_C" in name:
            contraste = "G vs C"
    elif "_F1RS_min_" in name:
        f1rs_type = "F1â€‘RS MÃ­nimo"
        if "G_vs_N" in name:
            contraste = "G vs N"
    elif "_ARS_max_" in name:  # Para compatibilidad con experimentos anteriores
        f1rs_type = "F1â€‘RS MÃ¡ximo"
        if "D_vs_notD" in name:
            contraste = "D vs Â¬D"
        elif "N_vs_All" in name:
            contraste = "N vs All"
    elif "_ARS_ctrl_" in name:  # Para compatibilidad
        f1rs_type = "F1â€‘RS Control"
        if "G_vs_C" in name:
            contraste = "G vs C"
    elif "_ARS_min_" in name:  # Para compatibilidad
        f1rs_type = "F1â€‘RS MÃ­nimo"
        if "G_vs_N" in name:
            contraste = "G vs N"

    return {
        "fase": fase,
        "dataset": dataset,
        "modelo": modelo,
        "f1rs_type": f1rs_type,
        "contraste": contraste,
        "folder_name": name
    }


def analyze_phase1(experiments_df):
    """Analiza experimentos de Fase 1 (Multi-etiqueta)"""
    phase1 = experiments_df[experiments_df['fase'] == 'Fase 1: Multi-etiqueta'].copy()

    if phase1.empty:
        print("No se encontraron experimentos de Fase 1")
        return

    print("\n" + "="*80)
    print("FASE 1: ANÃLISIS MULTI-ETIQUETA - DetecciÃ³n de Sesgos")
    print("="*80)

    for dataset in phase1['dataset'].unique():
        print(f"\n--- Dataset: {dataset} ---")
        subset = phase1[phase1['dataset'] == dataset]

        result_table = []
        for _, row in subset.iterrows():
            result_table.append({
                'Modelo': row['modelo'],
                'Accuracy': f"{row.get('accuracy', 0):.4f}",
                'F1 (Macro)': f"{row.get('f1_macro', 0):.4f}",
                'F1 (Micro)': f"{row.get('f1_micro', 0):.4f}",
                'AUC (Macro)': f"{row.get('auc_macro', 0):.4f}",
            })

        df_result = pd.DataFrame(result_table)
        print(df_result.to_string(index=False))


def analyze_phase3(experiments_df):
    """Analiza experimentos de Fase 3 y calcula âˆ†F1â€‘RS"""
    phase3 = experiments_df[experiments_df['fase'] == 'Fase 3: Contraste Binario'].copy()

    if phase3.empty:
        print("No se encontraron experimentos de Fase 3")
        return

    print("\n" + "="*80)
    print("FASE 3: CONTRASTES BINARIOS - CuantificaciÃ³n del Rango de Robustez")
    print("="*80)

    for dataset in phase3['dataset'].unique():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print('='*80)

        subset = phase3[phase3['dataset'] == dataset]

        for modelo in subset['modelo'].unique():
            model_data = subset[subset['modelo'] == modelo]

            print(f"\n--- {modelo} ---")

            # Construir tabla de resultados
            result_table = []
            f1rs_values = {}

            for _, row in model_data.iterrows():
                f1rs_type = row['f1rs_type']
                contraste = row['contraste']

                # F1â€‘RS = F1-score de la clase positiva
                f1rs_value = row.get('f1', row.get('f1_macro', 0))
                if f1rs_type:
                    f1rs_values[f1rs_type] = f1rs_value

                result_table.append({
                    'Tipo': f1rs_type if f1rs_type else "No especificado",
                    'Contraste': contraste if contraste else "No especificado",
                    'Precision': f"{row.get('precision', 0):.4f}",
                    'Recall': f"{row.get('recall', 0):.4f}",
                    'F1â€‘RS (F1)': f"{f1rs_value:.4f}",
                })

            df_result = pd.DataFrame(result_table)
            print(df_result.to_string(index=False))

            # Calcular âˆ†F1â€‘RS
            if 'F1â€‘RS MÃ¡ximo' in f1rs_values and 'F1â€‘RS MÃ­nimo' in f1rs_values:
                delta_f1rs = f1rs_values['F1â€‘RS MÃ¡ximo'] - f1rs_values['F1â€‘RS MÃ­nimo']
                print(f"\n{'â”€'*50}")
                print(f"âˆ†F1â€‘RS (Rango de Robustez): {delta_f1rs:.4f}")
                print(f"CaÃ­da de robustez: {(delta_f1rs / f1rs_values['F1â€‘RS MÃ¡ximo']) * 100:.2f}%")
                print(f"{'â”€'*50}")


def generate_comparison_table(experiments_df):
    """Genera tabla comparativa de todos los modelos"""
    phase3 = experiments_df[experiments_df['fase'] == 'Fase 3: Contraste Binario'].copy()

    if phase3.empty:
        return

    print("\n" + "="*80)
    print("TABLA COMPARATIVA: âˆ†F1â€‘RS POR MODELO Y DATASET")
    print("="*80 + "\n")

    comparison = []

    for dataset in phase3['dataset'].unique():
        subset = phase3[phase3['dataset'] == dataset]

        for modelo in subset['modelo'].unique():
            model_data = subset[subset['modelo'] == modelo]

            f1rs_max = model_data[model_data['f1rs_type'] == 'F1â€‘RS MÃ¡ximo']
            f1rs_min = model_data[model_data['f1rs_type'] == 'F1â€‘RS MÃ­nimo']

            if not f1rs_max.empty and not f1rs_min.empty:
                f1_max = f1rs_max.iloc[0].get('f1', f1rs_max.iloc[0].get('f1_macro', 0))
                f1_min = f1rs_min.iloc[0].get('f1', f1rs_min.iloc[0].get('f1_macro', 0))
                delta = f1_max - f1_min

                comparison.append({
                    'Dataset': dataset,
                    'Modelo': modelo,
                    'F1â€‘RS MÃ¡ximo': f"{f1_max:.4f}",
                    'F1â€‘RS MÃ­nimo': f"{f1_min:.4f}",
                    'âˆ†F1â€‘RS': f"{delta:.4f}",
                    'CaÃ­da (%)': f"{(delta/f1_max)*100:.2f}%"
                })

    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))

    # Guardar en CSV
    df_comparison.to_csv("f1rs_comparison_summary.csv", index=False)
    print("\n[INFO] Tabla comparativa guardada en: f1rs_comparison_summary.csv")


def main():
    print("\n" + "="*80)
    print(" "*20 + "ANÃLISIS DE RESULTADOS - MÃ‰TODO ASR")
    print("="*80)

    # Encontrar carpetas de experimentos
    folders = find_experiment_folders()

    if not folders:
        print("\n[ERROR] No se encontraron carpetas de experimentos (main_*)")
        return

    print(f"\n[INFO] Se encontraron {len(folders)} experimentos")

    # Recolectar datos
    experiments_data = []

    for folder in folders:
        category = categorize_experiment(folder)
        metrics = extract_test_metrics(folder)

        if metrics:
            experiments_data.append({**category, **metrics})
        else:
            print(f"[WARN] No se encontraron mÃ©tricas en: {folder.name}")

    if not experiments_data:
        print("\n[ERROR] No se pudieron extraer mÃ©tricas de ningÃºn experimento")
        return

    # Crear DataFrame
    df = pd.DataFrame(experiments_data)

    # AnÃ¡lisis por fases
    analyze_phase1(df)
    analyze_phase3(df)

    # Tabla comparativa final
    generate_comparison_table(df)

    print("\n" + "="*80)
    print("ANÃLISIS COMPLETADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()