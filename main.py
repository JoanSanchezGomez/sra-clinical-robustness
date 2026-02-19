# main.py
import os
import sys
import time
import numpy as np
import torch
import pandas as pd

# Importaciones para splits
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler

from codes.utils import parse_args, setup_output_folder
from codes.data_loader import CustomDataset
from codes.models import get_timm_model, save_model
from codes.training import train_one_epoch, evaluate
from codes.metrics import calculate_metrics, plot_training_history, plot_roc_curves
from codes.config import DATASETS


def main():

    # 1. Parámetros y configuración básica
    args = parse_args()
    np.random.seed(args.RAND_STATE)
    torch.manual_seed(args.RAND_STATE)

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    out_folder = setup_output_folder(script_name, args.IDENTIFIER, clean=args.CLEAN)
    print(f"[INFO] Carpeta de salida: {out_folder}")

    config_path = os.path.join(out_folder, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"[INFO] Configuración guardada en: {config_path}")



    # 2. Definir la configuración del dataset, la función de carga y el formato de etiquetas
    dataset_config = DATASETS[args.DATASET]
    load_func      = dataset_config['load_func']          # función de carga del dataset (todas las funciones creadas deben aceptar los mismos parámetros)
    data_folder    = dataset_config['data_folder']        # ruta al dataset
    label_format   = dataset_config['label_format']       # 'single' (una etiqueta por imagen) o 'multi' (varias etiquetas por imagen)
    labels_dict    = dataset_config['labels_dict']        # diccionario de etiquetas
    image_formats  = dataset_config['image_formats']      # tupla de extensiones de imagen incluidas en el dataset
    default_img_size = dataset_config['default_img_size'] # tamaño de imagen por defecto (si no se utilizan las transformaciones del preentrenamiento)



    # 3. Establecer la función de pérdida necesaria y el formato de etiquetas requerido para la configuración
    if args.POSITIVE_CLASS is not None: 
        # Si el problema de clasificación es binario (clase positiva/negativa), usamos BCEWithLogitsLoss con el formato de etiquetas 'class_index'
        loss_fn = BCEWithLogitsLoss()
        loss_label_format = 'bce'
    elif label_format == 'multi':
        # Si el problema es multi-label (varias etiquetas por imagen), usamos BCEWithLogitsLoss con el formato de etiquetas 'multi_hot'
        loss_fn = BCEWithLogitsLoss()
        loss_label_format = 'bce'
    elif label_format == 'single':
        # Si el problema es single-label (una etiqueta por imagen), usamos CrossEntropyLoss con el formato de etiquetas 'class_index'
        loss_fn = CrossEntropyLoss()
        loss_label_format = 'ce'
    


    # 4. Cargar el dataset y actualizar el diccionario de etiquetas (por si este hubiera cambiado)
    image_paths, labels_out, labels_dict = load_func(data_folder=data_folder, labels_dict=labels_dict, 
                                                     image_formats=image_formats, loss_label_format=loss_label_format, 
                                                     selected_labels=args.SELECTED_LABELS, positive_class=args.POSITIVE_CLASS)
    X_all = np.array(image_paths).reshape(-1, 1)
    y_all = labels_out



    # 5. División del dataset en train, val y test
    if label_format == 'multi':
        # Para multi-label, usamos iterative_train_test_split para mantener la distribución en cada etiqueta
        X_traval, y_traval, X_test, y_test = iterative_train_test_split(X_all, y_all, test_size=args.TEST_SIZE)
        X_train, y_train, X_val, y_val = iterative_train_test_split(X_traval, y_traval, test_size=args.VAL_SIZE)
    else:
        # Para single-label usamos train_test_split con estratificación
        if y_all.ndim == 2:
            stratify_all = np.argmax(y_all, axis=1) # si es one-hot, convertimos a índices ya que train_test_split espera enteros
        else:
            stratify_all = y_all.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=args.TEST_SIZE, random_state=args.RAND_STATE, stratify=stratify_all)
        if y_train.ndim == 2:
            stratify_train = np.argmax(y_train, axis=1)
        else:
            stratify_train = y_train.ravel()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.VAL_SIZE, random_state=args.RAND_STATE, stratify=stratify_train)

    # Convertir arrays de rutas en listas
    X_train = X_train.ravel().tolist()
    X_val   = X_val.ravel().tolist()
    X_test  = X_test.ravel().tolist()
    print(f"[INFO] División de datos: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")



    # 6. Determinar el numero neuronas de salida necesarias para el modelo de clasificación
    if y_all.ndim == 2 and y_all.shape[1] > 2: # formato one-hot o multi-hot
        n_classes = y_all.shape[1] 
    else: # formato de vector de enteros
        n_classes = len(np.unique(y_all)) 
    if n_classes==2: # clasificación binaria
        n_outs = 1
        print("[INFO] Clasificación binaria (una salida).")
    elif n_classes > 2: # clasificación multi-clase
        n_outs = n_classes
        print("[INFO] Clasificación multi-clase detectada (varias salidas).")
    else: # error en el número de clases
        raise ValueError(f"[ERROR] Número de clases no válido: {n_classes}. Debe ser mayor que 1.")



    # 7. Configuración del modelo
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    model, transforms = get_timm_model(args.MODEL_NAME, n_outs, args.PRETRAIN, args.FREEZE, default_img_size=default_img_size)
    model = model.to(device)
    print(f"[INFO] Modelo: {args.MODEL_NAME} con {n_outs} salidas")
    print(f"[INFO] Transformaciones: {transforms}")


    # —————— Balance de sampler ——————
    sampler   = None
    shuffle   = True

    # activamos sampler sólo en los casos “single-label”:
    #   • CrossEntropy (multi-clase single-label)
    #   • BCEWithLogitsLoss binario (n_outs == 1)
    if args.BALANCE_SAMPLER and (
        loss_label_format == 'ce' or
        (loss_label_format == 'bce' and n_outs == 1)
    ):

        # extraemos el índice de clase de cada muestra
        if loss_label_format == 'ce':
            # y_train puede ser one-hot o índices
            if y_train.ndim == 2:
                y_idx = np.argmax(y_train, axis=1)
            else:
                y_idx = y_train.ravel()
        else:
            # BCE binario: y_train es vector 0/1
            y_idx = y_train.ravel()

        y_idx = y_idx.astype(np.int64) # aseguramos que sea int64 para bincount
        class_counts   = np.bincount(y_idx, minlength=n_classes)
        class_weights  = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[y_idx]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # sampler ya baraja

    # —————— Balance de loss ——————
    if args.BALANCE_LOSS:
        if loss_label_format == 'ce':
            # CrossEntropy con weight
            if y_train.ndim == 2:
                counts = np.sum(y_train, axis=0)
            else:
                counts = np.bincount(y_train.ravel(), minlength=n_classes)
            weights = 1.0 / (counts + 1e-6)
            weights = weights * (n_classes / weights.sum())
            w_t = torch.tensor(weights, dtype=torch.float32).to(device)
            loss_fn = CrossEntropyLoss(weight=w_t)

        elif loss_label_format == 'bce':
            # BCE con pos_weight (funciona para binario y multi-label)
            pos = np.sum(y_train, axis=0)
            neg = y_train.shape[0] - pos
            pos_w = neg / (pos + 1e-6)
            w_t   = torch.tensor(pos_w, dtype=torch.float32).to(device)
            loss_fn = BCEWithLogitsLoss(pos_weight=w_t)

        else:
            print(f"[WARN] Balanceo de loss no soportado para '{loss_label_format}'")
    # —————————————————————————————



    # 8. Crear datasets, dataloaders y definir el optimizador
    train_loader = DataLoader(CustomDataset(X_train, y_train, n_outs, label_format, transforms), batch_size=args.BATCH_SIZE, sampler=sampler, shuffle=shuffle, num_workers=0)
    val_loader   = DataLoader(CustomDataset(X_val, y_val, n_outs, label_format, transforms), batch_size=args.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(CustomDataset(X_test, y_test, n_outs, label_format, transforms), batch_size=args.BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = Adam(model.parameters(), lr=args.LEARNING_RATE)



    # 9. Ciclo de entrenamiento
    metrics_history = []
    best_val_metric = -float('inf')
    best_model_state = None
    print("\n[INFO] Iniciando entrenamiento...")
    for epoch in range(args.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{args.EPOCHS} ---")
        start_time = time.time()

        train_loss, y_train_true, y_train_pred = train_one_epoch(
            model, train_loader, device, loss_fn, optimizer,
            label_format=label_format, n_outs=n_outs
        )
        val_loss, y_val_true, y_val_pred = evaluate(
            model, val_loader, device, loss_fn,
            label_format=label_format, n_outs=n_outs
        )
        
        # Se obtienen las métricas calculadas por calculate_metrics
        train_metrics = calculate_metrics(y_train_true, y_train_pred,
                                          label_format=label_format, n_outs=n_outs)
        val_metrics   = calculate_metrics(y_val_true, y_val_pred,
                                          label_format=label_format, n_outs=n_outs)

        epoch_time = time.time() - start_time
        print(f"Tiempo: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Mostrar Accuracy, Recall y F1 según el tipo de problema:
        if n_outs == 1:
            # Caso binario: se usan las métricas directas
            print(f"Train: Acc: {train_metrics['accuracy']:.4f} | Rec: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}")
            print(f"Val:   Acc: {val_metrics['accuracy']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            current_val_metric = val_metrics['f1']
        elif label_format == 'single':
            # Caso multi-clase single-label: se usan recall y f1 en promedio macro (accuracy global)
            print(f"Train: Acc: {train_metrics['accuracy']:.4f} | Rec (macro): {train_metrics['recall_macro']:.4f} | F1 (macro): {train_metrics['f1_macro']:.4f}")
            print(f"Val:   Acc: {val_metrics['accuracy']:.4f} | Rec (macro): {val_metrics['recall_macro']:.4f} | F1 (macro): {val_metrics['f1_macro']:.4f}")
            current_val_metric = val_metrics.get('f1_macro', 0)
        else:
            # Caso multi-label: se muestra la subset accuracy y las métricas micro
            print(f"Train: Subset Acc: {train_metrics['accuracy']:.4f} | Rec (micro): {train_metrics['recall_micro']:.4f} | F1 (micro): {train_metrics['f1_micro']:.4f}")
            print(f"Val:   Subset Acc: {val_metrics['accuracy']:.4f} | Rec (micro): {val_metrics['recall_micro']:.4f} | F1 (micro): {val_metrics['f1_micro']:.4f}")
            current_val_metric = val_metrics.get('f1_macro', 0)

        metrics_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        metrics_history.append(metrics_dict)

        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metric': current_val_metric
            }
            print(f"[INFO] Nuevo mejor modelo (F1: {current_val_metric:.4f})")
            save_model(model, optimizer, epoch + 1, {"val_f1": current_val_metric},
                       os.path.join(out_folder, "best_model.pth"))
            
    # Guardar modelo final
    save_model(model, optimizer, args.EPOCHS, {"val_f1": current_val_metric},
               os.path.join(out_folder, "final_model.pth"))

    if not args.USE_LAST and best_model_state:
        print(f"\n[INFO] Cargando mejor modelo (epoch {best_model_state['epoch']})")
        model.load_state_dict(best_model_state['model_state_dict'])

    pd.DataFrame(metrics_history).to_csv(os.path.join(out_folder, "metrics_history.csv"), index=False)
    plot_training_history(metrics_history, out_folder)



    # 10. Evaluación final en test
    print("\n--- Evaluación final en test ---")
    test_loss, y_test_true, y_test_pred = evaluate(
        model, test_loader, device, loss_fn,
        label_format=label_format, n_outs=n_outs
    )
    test_metrics = calculate_metrics(y_test_true, y_test_pred, label_format=label_format, n_outs=n_outs)
    
    # Guardar las predicciones y etiquetas verdaderas
    test_predictions_df = pd.DataFrame({"True_Label": y_test_true.tolist(), "Predicted_Label": y_test_pred.tolist()})
    test_predictions_df.to_csv(os.path.join(out_folder, "test_predictions.csv"), index=False)
    # Guardar las metricas en un CSV
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(os.path.join(out_folder, "test_metrics.csv"), index=False)

    print("\nResultados finales en test:")
    if n_outs == 1:
        print(f"Acc: {test_metrics['accuracy']:.4f} | Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    elif label_format == 'single':
        print(f"Acc: {test_metrics['accuracy']:.4f} | Rec (macro): {test_metrics['recall_macro']:.4f} | F1 (macro): {test_metrics['f1_macro']:.4f}")
    else:  # multi-label
        print(f"Subset Acc: {train_metrics['accuracy']:.4f} | Rec (micro): {train_metrics['recall_micro']:.4f} | F1 (micro): {train_metrics['f1_micro']:.4f}")
    
    if labels_dict is not None:
        plot_roc_curves(y_test_true, y_test_pred, labels_dict, label_format=label_format, n_outs=n_outs, save_folder=out_folder)

    print("\n[INFO] Proceso completado exitosamente.")





if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
