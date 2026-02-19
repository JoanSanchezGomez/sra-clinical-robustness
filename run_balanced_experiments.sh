#!/bin/bash

EPOCHS=50
BATCH_SIZE=32
LR=1e-4

echo "=================================================================="
echo "  EXPERIMENTOS CON BALANCEO - SOLO ODIR-5K"
echo "=================================================================="

# FASE 1: Multi-etiqueta
echo "[1/9] Fase 1 - Swin-Tiny"
python main.py -id _fase1_odir5k_multi_swin_tiny_BALANCED \
               -data odir5k -model swin_tiny_patch4_window7_224 \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[2/9] Fase 1 - ConvNeXt-Base"
python main.py -id _fase1_odir5k_multi_convnext_base_BALANCED \
               -data odir5k -model convnext_base \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[3/9] Fase 1 - EfficientNet-B0"
python main.py -id _fase1_odir5k_multi_efficientnet_b0_BALANCED \
               -data odir5k -model efficientnet_b0 \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

# FASE 3: Contrastes Swin-Tiny
echo "[4/9] Fase 3 - Swin ARS Máx (N vs All)"
python main.py -id _fase3_odir5k_swin_tiny_ARS_max_N_vs_All_BALANCED \
               -data odir5k -positive N -model swin_tiny_patch4_window7_224 \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[5/9] Fase 3 - Swin ARS Control (G vs C)"
python main.py -id _fase3_odir5k_swin_tiny_ARS_ctrl_G_vs_C_BALANCED \
               -data odir5k -selected G C -positive G \
               -model swin_tiny_patch4_window7_224 \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[6/9] Fase 3 - Swin ARS Mín (G vs N) - CRÍTICO"
python main.py -id _fase3_odir5k_swin_tiny_ARS_min_G_vs_N_BALANCED \
               -data odir5k -selected G N -positive G \
               -model swin_tiny_patch4_window7_224 \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

# FASE 3: Contrastes ConvNeXt-Base
echo "[7/9] Fase 3 - ConvNeXt ARS Máx (N vs All)"
python main.py -id _fase3_odir5k_convnext_base_ARS_max_N_vs_All_BALANCED \
               -data odir5k -positive N -model convnext_base \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[8/9] Fase 3 - ConvNeXt ARS Control (G vs C)"
python main.py -id _fase3_odir5k_convnext_base_ARS_ctrl_G_vs_C_BALANCED \
               -data odir5k -selected G C -positive G -model convnext_base \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "[9/9] Fase 3 - ConvNeXt ARS Mín (G vs N) - CRÍTICO"
python main.py -id _fase3_odir5k_convnext_base_ARS_min_G_vs_N_BALANCED \
               -data odir5k -selected G N -positive G -model convnext_base \
               -pretrain -freeze -epoch $EPOCHS -batch $BATCH_SIZE -lr $LR \
               -balance_loss

echo "=================================================================="
echo "  TODOS LOS EXPERIMENTOS COMPLETADOS"
echo "=================================================================="