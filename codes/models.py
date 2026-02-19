import torch
import timm
from timm.data import resolve_data_config, create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from codes.config import SUPPORTED_MODELS


def get_timm_model(name, num_classes, pretrained=True, freeze=True, default_img_size=(224, 224)):
    """
    Crea un modelo timm y, si se usa preentrenamiento, devuelve también
    las transformaciones compatibles con los pesos cargados.

    Returns:
        model (nn.Module): el modelo.
        transforms (callable or None): transformaciones si pretrained=True, si no, None.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo '{name}' no soportado.")

    model_name = SUPPORTED_MODELS[name]["pretrained_name"] if pretrained else SUPPORTED_MODELS[name]["default_name"]

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if freeze:
        # Congelar todos los parámetros
        for param in model.parameters():
            param.requires_grad = False
        # Identificar el clasificador dependiendo del modelo
        if hasattr(model, 'get_classifier'):
            classifier = model.get_classifier()
        else:
            raise ValueError("El modelo no tiene el método 'get_classifier' para obtener el clasificador.")
        # Descongelar los parámetros del clasificador
        for param in classifier.parameters():
            param.requires_grad = True

    transforms = None
    if pretrained:
        config = resolve_data_config({}, model=model)
        transforms = create_transform(**config)

    # Si no hay que aplicar transformaciones utilizadas en el preentrenamiento, se aplica una transformación por defecto para unificar los tamaños de imagen
    else:
        transforms = A.Compose([
            A.Resize(height=default_img_size[0], width=default_img_size[1]),
            ToTensorV2()
        ])

    return model, transforms
      
        

def save_model(model, optimizer, epoch, metrics, filename):
    """
    Guarda el modelo junto a los metadatos asociados.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"[INFO] Modelo guardado en: {filename}")
