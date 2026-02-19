import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             cohen_kappa_score, roc_curve)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------
# CÁLCULO DE MÉTRICAS
# Separamos la lógica en funciones auxiliares para cada tipo de problema.
# ---------------------------------------------

def _calculate_binary_metrics(y_true, y_pred):
    """Métricas para clasificación binaria (única salida)."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_bin),
        'precision': precision_score(y_true, y_pred_bin, zero_division=0),
        'recall': recall_score(y_true, y_pred_bin, zero_division=0),
        'f1': f1_score(y_true, y_pred_bin, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred_bin),
        'confusion_matrix': confusion_matrix(y_true, y_pred_bin).tolist(),
        'classification_report': classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['auc'] = float('nan')
    return metrics


def _calculate_multiclass_metrics(y_true, y_pred):
    """Métricas para clasificación multiclase (single-label)."""
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_classes),
        'precision_macro': precision_score(y_true, y_pred_classes, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_classes, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_classes, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred_classes, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_classes, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_classes, average='micro', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred_classes),
        'confusion_matrix': confusion_matrix(y_true, y_pred_classes).tolist(),
        'classification_report': classification_report(y_true, y_pred_classes, output_dict=True, zero_division=0)
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr')
    except ValueError:
        metrics['auc'] = float('nan')
    return metrics


def _calculate_multilabel_metrics(y_true, y_pred, n_outs):
    """Métricas para clasificación multi‑etiqueta."""
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_bin),
        'precision_micro': precision_score(y_true, y_pred_bin, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_bin, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_bin, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    }
    try:
        metrics['auc_micro'] = roc_auc_score(y_true, y_pred, average="micro")
        metrics['auc_macro'] = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        metrics['auc_micro'] = metrics['auc_macro'] = float('nan')

    # Calcular Cohen's Kappa por etiqueta y promediar
    kappas = [cohen_kappa_score(y_true[:, i], y_pred_bin[:, i]) for i in range(n_outs)]
    metrics['kappa'] = np.mean(kappas)
    
    # Matriz de confusión y classification report por etiqueta
    per_label_confusion = {}
    per_label_report = {}
    for i in range(n_outs):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        per_label_confusion[i] = cm.tolist()
        per_label_report[i] = classification_report(y_true[:, i], y_pred_bin[:, i], output_dict=True, zero_division=0)
    metrics['confusion_matrix_per_label'] = per_label_confusion
    metrics['classification_report_per_label'] = per_label_report

    return metrics


def calculate_metrics(y_true, y_pred, label_format='single', n_outs=2):
    """
    Calcula métricas de rendimiento en función del tipo de problema.
    
    Args:
        y_true (array): Etiquetas verdaderas.
        y_pred (array): Predicciones o probabilidades.
        label_format (str): 'single' (clasificación single-label) o 'multi' (multi-label).
        n_outs (int): Número de salidas o etiquetas.
    
    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    if n_outs == 1:
        return _calculate_binary_metrics(y_true, y_pred)
    elif label_format == 'single':
        return _calculate_multiclass_metrics(y_true, y_pred)
    else:  # Multi-label
        return _calculate_multilabel_metrics(y_true, y_pred, n_outs)


# ---------------------------------------------
# PLOTEO DE GRÁFICAS
# Funciones para dibujar la curva ROC, la matriz de confusión y el historial de entrenamiento.
# ---------------------------------------------

def plot_roc_curve_binary(y_true, y_preds, label_names, save_folder=None):
    """Dibuja la curva ROC para un problema binario (única salida)."""
    y_true = y_true.ravel()
    y_preds = y_preds.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_preds)
    auc_score = roc_auc_score(y_true, y_preds)

    # Como la lista de nombres de clase 'label_names' ya viene ordenada por índice de clase, la clase positiva es el índice 1
    positive_label_name = label_names[1] 

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{positive_label_name} (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    # plt.title("ROC Curve (Binary Classification)", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    if save_folder:
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "roc_curve_binary.png"))
    plt.close()
    return auc_score


def plot_confusion_matrix(cm, label_names, title="Confusion Matrix", save_path=None):
    """Dibuja y guarda la matriz de confusión."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(tick_marks, label_names, fontsize=12)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_roc_curves_multiclass(y_true, y_preds, label_names, n_outs, save_folder=None):
    """
    Dibuja las curvas ROC para cada clase en problemas multiclase o multi-label.
    
    Args:
        y_true (array): Etiquetas verdaderas en formato one-hot.
        y_preds (array): Predicciones/probabilidades con forma (N, n_outs).
        label_names (list): Lista de nombres de clases.
        n_outs (int): Número de clases/etiquetas.
        save_folder (str): Carpeta para guardar la gráfica (opcional).
    
    Returns:
        list: Lista de AUC para cada clase.
    """
    auc_scores = []
    plt.figure(figsize=(10, 8))
    for i in range(n_outs):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_preds[:, i])
            auc_score = roc_auc_score(y_true[:, i], y_preds[:, i])
            auc_scores.append(auc_score)
            plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC={auc_score:.2f})")
        except Exception as e:
            auc_scores.append(np.nan)
            print(f"Error calculando ROC para la clase {label_names[i]}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    # plt.title("ROC Curves (Multiclass Classification)", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    if save_folder:
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "roc_curves_multiclass.png"))
    plt.close()
    return auc_scores


def plot_roc_curves(y_true, y_preds, labels_dict, label_format='multi', n_outs=2, save_folder=None):
    """
    Función principal para dibujar las curvas ROC y, en su caso, la matriz de confusión.
    
    Args:
        y_true (array): Etiquetas verdaderas (vector o matriz).
        y_preds (array): Predicciones/probabilidades (forma: (N, n_outs)).
        labels_dict (dict): Diccionario con {índice: etiqueta o {'name': nombre}}.
        label_format (str): 'single' o 'multi'.
        n_outs (int): Número de salidas.
        save_folder (str): Carpeta para guardar las gráficas (opcional).
    
    Returns:
        list: Lista de AUC calculados.
    """
    # Extraer nombres de las clases ordenados por índice de clase en el diccionario (para que se muestren en el orden correcto)
    label_names = [
        v['name'] if isinstance(v, dict) else str(v)
        for _, v in sorted(labels_dict.items(), key=lambda item: item[1]['index'])
    ]


    # Caso binario: única salida
    if n_outs == 1:
        auc = plot_roc_curve_binary(y_true, y_preds, label_names, save_folder)
        pred_labels = (y_preds.ravel() >= 0.5).astype(int)
        cm = confusion_matrix(y_true.ravel(), pred_labels)
        if save_folder:
            plot_confusion_matrix(cm, label_names, title="",
                                  save_path=os.path.join(save_folder, "confusion_matrix_binary.png"))
        return [auc]
    else:
        # Convertir a one-hot si las etiquetas están en vector (para problemas single-label multiclase)
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_true = label_binarize(y_true.ravel(), classes=list(range(n_outs)))
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        aucs = plot_roc_curves_multiclass(y_true, y_preds, label_names, n_outs, save_folder)
        
        # Para single-label multiclase, dibujar la matriz de confusión
        if label_format == 'single' and n_outs > 1:
            true_labels = np.argmax(y_true, axis=1)
            pred_labels = np.argmax(y_preds, axis=1)
            cm = confusion_matrix(true_labels, pred_labels)
            if save_folder:
                plot_confusion_matrix(cm, label_names, title="",
                                      save_path=os.path.join(save_folder, "confusion_matrix_multiclass.png"))
        return aucs


def plot_metric(df, x, ys, labels, title, ylabel, save_path):
    """Dibuja una métrica (o varias) a lo largo del entrenamiento."""
    plt.figure(figsize=(8, 6))
    for col, label in zip(ys, labels):
        plt.plot(df[x], df[col], label=label)
    plt.xlabel(x.capitalize(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(metrics_history, out_folder):
    """Dibuja y guarda las curvas de pérdida, F1 y accuracy a lo largo del entrenamiento."""
    metrics_df = pd.DataFrame(metrics_history)

    # Historial de Loss
    plot_metric(
        metrics_df,
        x='epoch',
        ys=['train_loss', 'val_loss'],
        labels=['Train Loss', 'Val Loss'],
        title='',
        ylabel='Loss',
        save_path=os.path.join(out_folder, "loss_history.png")
    )

    # Historial de F1-Score (si existen las columnas)
    if 'train_f1_macro' in metrics_df and 'train_f1_micro' in metrics_df:
        plot_metric(
            metrics_df,
            x='epoch',
            ys=['train_f1_macro', 'val_f1_macro', 'train_f1_micro', 'val_f1_micro'],
            labels=['Train F1 Macro', 'Val F1 Macro', 'Train F1 Micro', 'Val F1 Micro'],
            title='',
            ylabel='F1-score',
            save_path=os.path.join(out_folder, "f1_history.png")
        )

    # Historial de Accuracy (si existen las columnas)
    if 'train_accuracy' in metrics_df and 'val_accuracy' in metrics_df:
        plot_metric(
            metrics_df,
            x='epoch',
            ys=['train_accuracy', 'val_accuracy'],
            labels=['Train Accuracy', 'Val Accuracy'],
            title='',
            ylabel='Accuracy',
            save_path=os.path.join(out_folder, "accuracy_history.png")
        )

