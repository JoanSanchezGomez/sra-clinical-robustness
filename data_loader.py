import os
import warnings
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from codes.config import ODIR5K_LABELS


def load_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, n_outs, label_format, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.n_outs = n_outs
        self.label_format = label_format
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_img(self.image_paths[idx])

        if self.transforms:
            if isinstance(self.transforms, A.BasicTransform) or isinstance(self.transforms, A.Compose):
                # Transform de albumentations
                img = self.transforms(image=img)['image']
            else:
                # Transform de timm / torchvision
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = self.transforms(img)
        # Si no hay transformaciones, simplemente reordenar canales y convertir a tensor
        else: 
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        label = self.labels[idx]
        if self.label_format == "multi" or self.n_outs == 1:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return img, label





def load_edc(*, data_folder, labels_dict, image_formats, loss_label_format='bce',
             selected_labels=None, positive_class=None):
    """
    Carga imágenes del dataset EDC con etiquetas en el formato deseado.

    Args:
        data_folder (str): Ruta al dataset.
        labels_dict (dict): Diccionario de etiquetas original, formato:
            {'directorio': {'index': int, 'name': str}, ...}
        image_format (tuple[str]): Tupla de extensiones de imagen a cargar.
        loss_label_format (str): 'bce' -> (batch, 1) float, 'ce' -> (batch,) int64.
        selected_labels (list[str], opcional): Subconjunto de clases a cargar.
        positive_class (str, opcional): Clave del diccionario para generar problema binario.

    Returns:
        image_paths (list[str]): Rutas de imágenes.
        labels_array (np.ndarray): Etiquetas según formato solicitado.
        new_labels_dict (dict): Diccionario actualizado acorde a las etiquetas reales.
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"No existe: {data_folder}")

    print(f"[INFO] Cargando datos desde '{data_folder}'...")

    # Si no se especifican etiquetas, usar todas
    if selected_labels is None:
        selected_labels = list(labels_dict.keys())

    # Filtrar diccionario original según clases seleccionadas
    selected_dict = {k: labels_dict[k] for k in selected_labels if k in labels_dict}

    image_paths, labels = [], []

    # Iterar sobre el diccionario filtrado
    for cls_key, cls_info in tqdm(selected_dict.items(), desc="Procesando clases", dynamic_ncols=True, smoothing=0.1):
        cls_folder = os.path.join(data_folder, cls_key)
        if not os.path.exists(cls_folder):
            print(f"[WARNING] Carpeta no encontrada: '{cls_folder}', se omite.")
            continue

        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(image_formats):
                image_paths.append(os.path.join(cls_folder, fname))
                labels.append(cls_info['index'])

    if not image_paths:
        raise ValueError("No se han encontrado imágenes con etiquetas válidas.")

    labels_array = np.array(labels) # (num_samples,)
    new_labels_dict = selected_dict.copy()

    # Conversión a problema de clasificación binario
    if positive_class is not None:
        if positive_class not in selected_dict:
            raise ValueError(f"'{positive_class}' no es una clase válida. Debe estar en {list(selected_dict.keys())}")
        
        positive_idx = selected_dict[positive_class]['index']
        labels_array = np.array(labels_array == positive_idx, dtype=np.int64) # (num_samples,)
        print(f"[INFO] Problema binario creado: clase positiva '{positive_class}' (índice {positive_idx})")

        # Actualizar diccionario a clases binarias
        new_labels_dict = {
            'other': {'index': 0, 'name': 'Other'},
            positive_class: {'index': 1, 'name': selected_dict[positive_class]['name']}
        }
        print(f"[INFO] Clases generadas: {new_labels_dict}")

    # Conversión de las etiquetas al formado esperado por la loss
    if loss_label_format == "bce":
        if len(new_labels_dict) == 2: # (binary)
            labels_array = np.array(labels_array, dtype=float).reshape(-1, 1) # (num_samples, ) -> (num_samples, 1) float
        elif len(new_labels_dict) > 2: # (multi-class)
            num_classes = len(new_labels_dict)
            labels_array = np.eye(num_classes, dtype=np.float)[labels_array] # (num_samples, ) -> (num_samples, num_classes) float
        else:
            raise ValueError(f"El número de clases no puede ser inferior a dos: {len(new_labels_dict)} clases encontradas.")
    
    elif loss_label_format == "ce": # (single-class) 
        labels_array = np.array(labels_array, dtype=np.int64) # (num_samples, ) -> (num_samples, ) int64
    
    else:
        raise ValueError(f"Formato de pérdida no soportado: '{loss_label_format}'. Debe ser 'bce' o 'ce'.")

    print(f"[INFO] Cargadas {len(image_paths)} imágenes de {len(new_labels_dict)} clases.")

    return image_paths, labels_array, new_labels_dict






def load_odir5k(*,
                data_folder: str,
                labels_dict: dict,
                image_formats: tuple,
                loss_label_format: str = 'bce',
                selected_labels: list = None,
                positive_class: str = None):
    """
    Carga las rutas de las imágenes del dataset ODIR-5K y sus etiquetas
    en el formato pedido (bce o ce), permitiendo también convertir a binario.

    Args:
        data_folder: ruta al directorio que contiene
            - 'Training Images/' con las imágenes
            - 'data.xlsx' con las etiquetas
        labels_dict: dict con mapeo {col_name: {'index': int, 'name': str}, ...}
        image_formats: tupla de extensiones válidas ('.png', '.jpg', ...)
        loss_label_format: 'bce' para multi-hot floats, 'ce' para índices int64
        selected_labels: lista de columnas del Excel a usar (por defecto todas)
        positive_class: si no es None, crea problema binario usando esa etiqueta

    Returns:
        image_paths: list[str]
        labels_array: np.ndarray de forma
            - (N, 1) float32 si binario + bce
            - (N, C) float32 si multi-etiqueta + bce
            - (N,)   int64  si ce
        new_labels_dict: dict actualizado con índices consecutivos
    """
    image_folder = os.path.join(data_folder, 'Training Images')
    label_file   = os.path.join(data_folder, 'data.xlsx')

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"No existe carpeta de imágenes: {image_folder}")
    if not os.path.isfile(label_file):
        raise FileNotFoundError(f"No existe archivo de etiquetas: {label_file}")

    print(f"[INFO] Cargando imágenes de '{image_folder}' y etiquetas de '{label_file}'...")
    # Leer Excel con pandas
    df = pd.read_excel(label_file, engine="openpyxl")

    # Determinar qué etiquetas usar
    if selected_labels is None:
        selected_labels = list(labels_dict.keys())
    # Filtrar labels_dict a solo las seleccionadas
    selected_dict = {k: labels_dict[k] for k in selected_labels if k in labels_dict}
    if not selected_dict:
        raise ValueError(f"No hay claves válidas en selected_labels: {selected_labels}")

    # Recorrer todas las imágenes y emparejar con su fila de etiquetas
    file_names = [fn for fn in os.listdir(image_folder)
                  if fn.lower().endswith(image_formats)]
    image_paths = []
    label_rows  = []
    for fn in tqdm(file_names, desc="Procesando imágenes", dynamic_ncols=True, smoothing=0.1):
        img_path = os.path.join(image_folder, fn)
        try:
            ID = int(fn.split('_')[0])
        except ValueError:
            warnings.warn(f"Nombre inesperado '{fn}', se omite", UserWarning)
            continue

        # Extraer la fila correspondiente
        row = df[df['ID'] == ID]
        if row.empty:
            warnings.warn(f"No hay etiquetas para '{fn}', se omite", UserWarning)
            continue

        image_paths.append(img_path)
        label_rows.append(row[selected_labels])  # solo columnas seleccionadas

    if not image_paths:
        raise ValueError("No se encontraron imágenes con etiquetas válidas.")

    # Concatenar filas en un solo DataFrame
    df_labels = pd.concat(label_rows, ignore_index=True)
    # Obtener matriz numpy multi-hot
    labels_array = df_labels.values.astype(np.float32)  # (N, C)

    # Reconstruir dict con índices 0..C-1 según order de selected_labels
    new_labels_dict = {
        cls: {'index': i, 'name': selected_dict[cls]['name']}
        for i, cls in enumerate(selected_labels)
    }

    # Si piden problema binario
    if positive_class is not None:
        if positive_class not in new_labels_dict:
            raise ValueError(f"'{positive_class}' no está en selected_labels")
        pos_idx = new_labels_dict[positive_class]['index']
        # convertir multi-hot a 0/1 en base a esa columna
        labels_array = (labels_array[:, pos_idx] > 0).astype(np.int64)  # (N,)
        print(f"[INFO] Binario: positiva '{positive_class}' -> índice {pos_idx}")
        # reconstruir dict binario
        new_labels_dict = {
            'other':          {'index': 0, 'name': 'Other'},
            positive_class:  {'index': 1, 'name': new_labels_dict[positive_class]['name']}
        }

    # Ajustar al formato de loss pedido
    if loss_label_format == 'bce':
        if len(new_labels_dict) == 2:
            # binario: (N,) -> (N,1) float
            labels_array = labels_array.astype(np.float32).reshape(-1, 1)
        else:
            # multi-etiqueta: ya es multi-hot float32
            pass

    elif loss_label_format == 'ce':
        # necesita (N,) int64 de índices
        if labels_array.ndim == 2:
            # tomar argmax asumiendo una sola etiqueta activa por muestra
            labels_array = np.argmax(labels_array, axis=1)
        labels_array = labels_array.astype(np.int64)

    else:
        raise ValueError(f"Formato de pérdida no soportado: '{loss_label_format}'")

    print(f"[INFO] Cargadas {len(image_paths)} imágenes con {len(new_labels_dict)} clases.")
    return image_paths, labels_array, new_labels_dict



