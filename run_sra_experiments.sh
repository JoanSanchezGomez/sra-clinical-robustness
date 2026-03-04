#!/bin/bash

# Script to run SRA Method experiments
# Updated version with Swin Transformer
# 3 Models: EfficientNet-B0, ConvNeXt-Base, Swin-Tiny

MAIN="main.py"
if [ ! -f "$MAIN" ]; then
    echo "Error: $MAIN not found!"
    exit 1
fi

echo "================================================================================"
echo "           SRA METHOD - SYSTEMIC RELATIONSHIP ANALYSIS"
echo "         3 Models: EfficientNet-B0 | ConvNeXt-Base | Swin-Tiny"
echo "================================================================================"
echo ""

####################################
# GLOBAL CONFIGURATION
####################################
EPOCHS=50
BATCH_SIZE=32
LR=1e-4

echo "Training configuration:"
echo "  - Epochs:        $EPOCHS"
echo "  - Batch size:    $BATCH_SIZE"
echo "  - Learning rate: $LR"
echo ""

####################################
# SRA METHOD EXPERIMENTS
####################################

declare -a EXPERIMENTS=(
    #========================================
    # PHASE 1: MULTI-LABEL ANALYSIS (EDC)
    # Bias and correlation detection
    #========================================

    # EfficientNet-B0 - Lightweight efficient CNN
    "-id _phase1_edc_multi_efficientnet_b0 -data edc -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ConvNeXt-Base - Modern robust CNN
    "-id _phase1_edc_multi_convnext_base -data edc -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # Swin Transformer Tiny - Hierarchical Transformer
    "-id _phase1_edc_multi_swin_tiny -data edc -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    #========================================
    # PHASE 3: BINARY CONTRASTS (EDC)
    # Robustness Range quantification (ΔF1-RS)
    #========================================

    # --- EfficientNet-B0 ---
    # F1-RS Maximum: D vs ¬D (optimal discrimination contrast)
    "-id _phase3_edc_efficientnet_b0_SRA_max_D_vs_notD -data edc -positive diabetic_retinopathy -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # F1-RS Minimum: G vs N (maximum clinical confusion contrast)
    "-id _phase3_edc_efficientnet_b0_SRA_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # --- ConvNeXt-Base ---
    # F1-RS Maximum: D vs ¬D
    "-id _phase3_edc_convnext_base_SRA_max_D_vs_notD -data edc -positive diabetic_retinopathy -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # F1-RS Minimum: G vs N (CRITICAL)
    "-id _phase3_edc_convnext_base_SRA_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # --- Swin Transformer Tiny ---
    # F1-RS Maximum: D vs ¬D
    "-id _phase3_edc_swin_tiny_SRA_max_D_vs_notD -data edc -positive diabetic_retinopathy -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    # F1-RS Minimum: G vs N (CRITICAL)
    "-id _phase3_edc_swin_tiny_SRA_min_G_vs_N -data edc -selected glaucoma normal -positive glaucoma -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"


    #========================================
    # VALIDATION ON ODIR-5K
    # Realistic multi-label clinical setting
    #========================================

    # Phase 1: Multi-label ODIR-5K
    "-id _phase1_odir5k_multi_efficientnet_b0 -data odir5k -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _phase1_odir5k_multi_convnext_base -data odir5k -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _phase1_odir5k_multi_swin_tiny -data odir5k -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # Phase 3: Critical contrasts ODIR-5K

    # EfficientNet-B0
    "-id _phase3_odir5k_efficientnet_b0_SRA_max_N_vs_All -data odir5k -positive N -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _phase3_odir5k_efficientnet_b0_SRA_min_G_vs_N -data odir5k -selected G N -positive G -model efficientnet_b0 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # ConvNeXt-Base
    "-id _phase3_odir5k_convnext_base_SRA_max_N_vs_All -data odir5k -positive N -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _phase3_odir5k_convnext_base_SRA_min_G_vs_N -data odir5k -selected G N -positive G -model convnext_base -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"

    # Swin Transformer Tiny
    "-id _phase3_odir5k_swin_tiny_SRA_max_N_vs_All -data odir5k -positive N -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
    "-id _phase3_odir5k_swin_tiny_SRA_min_G_vs_N -data odir5k -selected G N -positive G -model swin_tiny_patch4_window7_224 -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR"
)

####################################
# EXECUTION
####################################

SHOW_OUTPUT=1 # Set to 0 to suppress real-time output

if [ "$SHOW_OUTPUT" -eq 1 ]; then
    echo "[INFO] Experiment output will be shown in real time..."
else
    echo "[INFO] Output will be saved to log files only..."
fi

echo ""
echo "[INFO] Total experiments to run: ${#EXPERIMENTS[@]}"
echo ""
echo "Breakdown:"
echo "  - Phase 1 EDC:      3 experiments (multi-label)"
echo "  - Phase 3 EDC:      6 experiments (3 models x 2 contrasts)"
echo "  - Phase 1 ODIR-5K:  3 experiments (multi-label)"
echo "  - Phase 3 ODIR-5K:  6 experiments (3 models x 2 contrasts)"
echo ""
read -p "Continue? (press Enter or wait 5 seconds)" -t 5
echo ""

COUNTER=1
FAILED=0
SUCCESS=0

for EXP in "${EXPERIMENTS[@]}"; do
    ID=$(echo "$EXP" | grep -oP '(?<=-id )[^ ]+')
    LOGFILE="${ID}.log"

    echo "================================================================================"
    echo "  Experiment [$COUNTER/${#EXPERIMENTS[@]}]: $ID"
    echo "================================================================================"

    if [ "$SHOW_OUTPUT" -eq 1 ]; then
        python "$MAIN" $EXP 2>&1 | tee "$LOGFILE"
    else
        python "$MAIN" $EXP > "$LOGFILE" 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "[✓] Experiment $ID completed successfully"
        ((SUCCESS++))
    else
        echo "[✗] ERROR in experiment $ID - See log: $LOGFILE"
        ((FAILED++))
    fi

    echo ""
    ((COUNTER++))
done

echo "================================================================================"
echo "                        EXECUTION SUMMARY"
echo "================================================================================"
echo ""
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "  ✓ Successful: $SUCCESS"
echo "  ✗ Failed:     $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
else
    echo "WARNING: Some experiments failed."
    echo "Check the .log files for details."
fi

echo ""
echo "Results saved in folders: main_*"
echo ""
echo "Next step:"
echo "  python analyze_sra_results.py"
echo ""
echo "================================================================================"
