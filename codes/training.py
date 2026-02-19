import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, device, loss_fn, optimizer, label_format='single', n_outs=1):
    """
    Realiza una época de entrenamiento.
    
    Args:
        model: El modelo de PyTorch.
        loader: Dataloader con batches (imágenes, etiquetas).
        device: Dispositivo (cpu o cuda).
        loss_fn: Función de pérdida.
        optimizer: Optimizador.
        label_format (str): 'single' para single-label o 'multi' para multi-label.
        n_outs (int): Número de salidas. Si es 1 se entiende que es clasificación binaria (una única salida).
    
    Returns:
        avg_loss: Pérdida promedio por batch.
        y_true: Etiquetas verdaderas (numpy array).
        y_pred: Probabilidades/predicciones (numpy array).
    """
    model.train()
    epoch_loss = 0
    y_pred_list, y_true_list = [], []
    
    for images, labels in tqdm(loader, desc="Entrenando", leave=False, dynamic_ncols=True, smoothing=0.1, file=sys.stderr):
                    
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Seleccionar activación:
        # - En problemas multi-label (label_format='multi') se usa sigmoid.
        # - En single-label se distingue según n_outs: 
        #    * Si n_outs == 1 (clasificación binaria con única salida): sigmoid.
        #    * Si n_outs > 1 (multi-clase): softmax.
        if n_outs == 1 or label_format == 'multi':
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
        else:
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()

        y_pred_list.append(probs)
        y_true_list.append(labels.cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    if label_format == 'single' and n_outs > 1:
        # Para multi-clase single-label se obtiene un vector unidimensional para las etiquetas.
        y_true = np.concatenate(y_true_list)
    else:
        y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return avg_loss, y_true, y_pred


def evaluate(model, loader, device, loss_fn, label_format='single', n_outs=1):
    """
    Evalúa el modelo en un conjunto de datos.
    
    Args:
        model: El modelo.
        loader: Dataloader con batches (imágenes, etiquetas).
        device: Dispositivo.
        loss_fn: Función de pérdida.
        label_format (str): 'single' para single-label o 'multi' para multi-label.
        n_outs (int): Número de salidas.
    
    Returns:
        avg_loss: Pérdida promedio.
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones/probabilidades.
    """
    model.eval()
    epoch_loss = 0
    y_pred_list, y_true_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluando", leave=False, dynamic_ncols=True, smoothing=0.1, file=sys.stderr):
            # En clasificación single-label con única salida (binary), aseguramos que labels tenga forma [batch, 1].
            if label_format == 'single' and n_outs == 1 and labels.dim() == 1:
                labels = labels.unsqueeze(1)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            
            if label_format == 'single':
                if n_outs == 1:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
            else:  # label_format == 'multi'
                probs = torch.sigmoid(outputs).cpu().numpy()
                
            y_pred_list.append(probs)
            y_true_list.append(labels.cpu().numpy())
    
    avg_loss = epoch_loss / len(loader)
    if label_format == 'single' and n_outs > 1:
        y_true = np.concatenate(y_true_list)
    else:
        y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return avg_loss, y_true, y_pred
