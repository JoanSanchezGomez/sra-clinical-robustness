import argparse
import os
import shutil
from codes.config import SUPPORTED_MODELS, DATASETS


def parse_args():
    """Define y parsea los argumentos del programa."""
    parser = argparse.ArgumentParser(description='Entrenamiento y evaluación para ODIR-5K usando timm')
    parser.add_argument('-id', '--identifier', dest='IDENTIFIER', default="", type=str, help='Identificador para la carpeta de salida')
    parser.add_argument('-clean', '--clean-files', dest='CLEAN', action="store_true", help='Eliminar archivos existentes en la carpeta de salida')
    parser.add_argument('-data', '--dataset', dest='DATASET', default='ucmerced', type=str, choices=DATASETS, help='Directorio del dataset a utilizar')
    parser.add_argument('-positive', '--positive-class', dest='POSITIVE_CLASS', type=str, default=None, help='Etiqueta de la clase positiva para clasificación binaria')    
    parser.add_argument('-selected', '--selected-labels', dest='SELECTED_LABELS', type=str, nargs='+', default=None, help='Lista de etiquetas a seleccionar')
    parser.add_argument('-rand', '--random-state', dest='RAND_STATE', default=42, type=int, help='Semilla aleatoria para reproducibilidad')
    parser.add_argument('-tst', '--test-size', dest='TEST_SIZE', default=0.2, type=float, help='Fracción de datos para test')
    parser.add_argument('-val', '--val-size', dest='VAL_SIZE', default=0.2, type=float, help='Fracción de training para validación')
    parser.add_argument('-model', '--model-name', dest='MODEL_NAME', default='resnet18', type=str, choices=SUPPORTED_MODELS, help='Nombre del modelo a usar')
    parser.add_argument('-pretrain', '--pretrained-weights', dest='PRETRAIN', action="store_true", help='Usar pesos preentrenados de image-net')
    parser.add_argument('-freeze', '--freeze-weights', dest='FREEZE', action="store_true", help='Se congelan todas las capas excepto el clasificador')
    parser.add_argument('-batch', '--batch-size', dest='BATCH_SIZE', default=32, type=int, help='Tamaño del batch de entrenamiento')
    parser.add_argument('-epoch', '--number-epochs', dest='EPOCHS', default=10, type=int, help='Número de épocas de entrenamiento')
    parser.add_argument('-lr', '--learning-rate', dest='LEARNING_RATE', default=1e-3, type=float, help='Tasa de aprendizaje')
    parser.add_argument('-gpu', '--gpu', dest='GPU', default=0, type=int, help='ID de GPU a usar')
    parser.add_argument('-last', '--use-last-epoch', dest='USE_LAST', action="store_true", help='Usar último modelo en lugar del mejor')
    parser.add_argument('-balance_loss', '--balance-loss', dest='BALANCE_LOSS', action='store_true', help='Usar pesos en la función de pérdida para balancear las clases')
    parser.add_argument('-balance_sampler', '--balance-sampler', dest='BALANCE_SAMPLER', action='store_true', help='Usar un sampler ponderado para el dataloader de entrenamiento')   
    args = parser.parse_args()
    
    if args.FREEZE and not args.PRETRAIN:
        parser.error("La opción --freeze-weights solo puede usarse si se activan los pesos preentrenados con --pretrained-weights.")

    if args.BALANCE_LOSS and args.BALANCE_SAMPLER:
        parser.warning("Ambas opciones --balance-loss y --balance-sampler están activadas. Se recomienda usar solo una de ellas para evitar sobrecompensación.")
    
    return args


def setup_output_folder(script_name, identifier, clean=False):
    """
    Crea la carpeta de salida.
    """
    out_folder = script_name + identifier
    if os.path.isdir(out_folder) and clean:
        shutil.rmtree(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    return out_folder

