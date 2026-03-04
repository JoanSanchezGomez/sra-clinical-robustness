#!/bin/bash

# Script para ejecutar experimentos del Método ASR
# Versión actualizada con Swin Transformer
# 3 Modelos: EfficientNet-B0, ConvNeXt-Base, Swin-Tiny

MAIN="main.py"
if [ ! -f "$MAIN" ]; then
    echo "Error: $MAIN not found!"
    exit 1
fi

echo "================================================================================"
echo "           MÉTODO ASR - ANÁLISIS SISTÉMICO DE RELACIONES"
echo "         3 Modelos: EfficientNet-B0 | ConvNeXt-Base | Swin-Tiny"
echo "================================================================================"
echo ""

####################################
# CONFIGURACIÓN GLOBAL
####################################
EPOCHS=50
BATCH_SIZE=32
LR=1e-4

echo "Configuración de entrenamiento:"
echo "  - Épocas: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LR"
echo ""

####################################
# EXPERIMENTOS MÉTODO ASR
####################################

declare -a EXPERIMENTS=(
    #========================================
    # FASE 1: ANÁLISIS MULTI-ETIQUETA (EDC)
    # Detección de sesgos y correlaciones
    #========================================

    # EfficientNet-B0 - CNN ligero eficiente
    "-id _fase1_edc_multi_efficientnet_b0 -data edc -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ConvNeXt-Base - CNN moderno robusto
    "-id _fase1_edc_multi_convnext_base -data edc -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # Swin Transformer Tiny - Transformer jerárquico
    "-id _fase1_edc_multi_swin_tiny -data edc -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    #========================================
    # FASE 3: CONTRASTES BINARIOS (EDC)
    # Cuantificación del Rango de Robustez (∆ARS)
    #========================================

    # --- EfficientNet-B0 ---
    # ARS Máximo: D vs ¬D (baseline - contraste más fácil)
    "-id _fase3_edc_efficientnet_b0_ARS_max_D_vs_notD -data edc -positive diabetic_retinopathy -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Control: G vs C (control intermedio)
    "-id _fase3_edc_efficientnet_b0_ARS_ctrl_G_vs_C -data edc -selected glaucoma cataract -positive glaucoma -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Mínimo: G vs N (contraste crítico)
    "-id _fase3_edc_efficientnet_b0_ARS_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # --- ConvNeXt-Base ---
    # ARS Máximo: D vs ¬D
    "-id _fase3_edc_convnext_base_ARS_max_D_vs_notD -data edc -positive diabetic_retinopathy -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Control: G vs C
    "-id _fase3_edc_convnext_base_ARS_ctrl_G_vs_C -data edc -selected glaucoma cataract -positive glaucoma -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Mínimo: G vs N (CRÍTICO)
    "-id _fase3_edc_convnext_base_ARS_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # --- Swin Transformer Tiny ---
    # ARS Máximo: D vs ¬D
    "-id _fase3_edc_swin_tiny_ARS_max_D_vs_notD -data edc -positive diabetic_retinopathy -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Control: G vs C
    "-id _fase3_edc_swin_tiny_ARS_ctrl_G_vs_C -data edc -selected glaucoma cataract -positive glaucoma -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ARS Mínimo: G vs N (CRÍTICO)
    "-id _fase3_edc_swin_tiny_ARS_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    #========================================
    # VALIDACIÓN EN ODIR-5K (OPCIONAL)
    # Entorno clínico realista multi-etiqueta
    #========================================

    # FASE 1: Multi-etiqueta ODIR-5K
    "-id _fase1_odir5k_multi_efficientnet_b0 -data odir5k -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase1_odir5k_multi_convnext_base -data odir5k -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase1_odir5k_multi_swin_tiny -data odir5k -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # FASE 3: Contrastes críticos ODIR-5K

    # EfficientNet-B0
    "-id _fase3_odir5k_efficientnet_b0_ARS_max_N_vs_All -data odir5k -positive N -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_efficientnet_b0_ARS_ctrl_G_vs_C -data odir5k -selected G C -positive G -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_efficientnet_b0_ARS_min_G_vs_N -data odir5k -selected G N -positive G -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ConvNeXt-Base
    "-id _fase3_odir5k_convnext_base_ARS_max_N_vs_All -data odir5k -positive N -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_convnext_base_ARS_ctrl_G_vs_C -data odir5k -selected G C -positive G -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_convnext_base_ARS_min_G_vs_N -data odir5k -selected G N -positive G -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # Swin Transformer Tiny
    "-id _fase3_odir5k_swin_tiny_ARS_max_N_vs_All -data odir5k -positive N -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_swin_tiny_ARS_ctrl_G_vs_C -data odir5k -selected G C -positive G -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _fase3_odir5k_swin_tiny_ARS_min_G_vs_N -data odir5k -selected G N -positive G -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
)

####################################
# EJECUCIÓN
####################################

SHOW_OUTPUT=1 # Cambiar a 0 para silenciar salida en tiempo real

if [ "$SHOW_OUTPUT" -eq 1 ]; then
    echo "[INFO] Se mostrará la salida de los experimentos en tiempo real..."
else
    echo "[INFO] La salida se guardará solo en logs..."
fi

echo ""
echo "[INFO] Total de experimentos a ejecutar: ${#EXPERIMENTS[@]}"
echo ""
echo "Desglose:"
echo "  - Fase 1 EDC:      3 experimentos (multi-etiqueta)"
echo "  - Fase 3 EDC:      9 experimentos (3 modelos × 3 contrastes)"
echo "  - Fase 1 ODIR-5K:  3 experimentos (multi-etiqueta)"
echo "  - Fase 3 ODIR-5K:  9 experimentos (3 modelos × 3 contrastes)"
echo ""
read -p "¿Continuar? (presiona Enter o espera 5 segundos)" -t 5
echo ""

COUNTER=1
FAILED=0
SUCCESS=0

for EXP in "${EXPERIMENTS[@]}"; do
    ID=$(echo "$EXP" | grep -oP '(?<=-id )[^ ]+')
    LOGFILE="${ID}.log"

    echo "================================================================================"
    echo "  Experimento [$COUNTER/${#EXPERIMENTS[@]}]: $ID"
    echo "================================================================================"

    if [ "$SHOW_OUTPUT" -eq 1 ]; then
        python "$MAIN" $EXP 2>&1 | tee "$LOGFILE"
    else
        python "$MAIN" $EXP > "$LOGFILE" 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "[✓] Experimento $ID completado exitosamente"
        ((SUCCESS++))
    else
        echo "[✗] ERROR en experimento $ID - Ver log: $LOGFILE"
        ((FAILED++))
    fi

    echo ""
    ((COUNTER++))
done

echo "================================================================================"
echo "                    RESUMEN DE EJECUCIÓN"
echo "================================================================================"
echo ""
echo "Total experimentos: ${#EXPERIMENTS[@]}"
echo "  ✓ Exitosos: $SUCCESS"
echo "  ✗ Fallidos:  $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "¡TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE!"
else
    echo "ATENCIÓN: Algunos experimentos fallaron."
    echo "Revisa los archivos .log para más detalles"
fi

echo ""
echo "Resultados guardados en carpetas: main_*"
echo ""
echo "Siguiente paso:"
echo "  python analyze_asr_results.py"
echo ""
echo "================================================================================"
