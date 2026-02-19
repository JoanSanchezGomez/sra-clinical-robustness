# Listado de modelos soportados
SUPPORTED_MODELS = {
    # Modelos clásicos y ligeros
    "resnet18": {
        "pretrained_name": "resnet18.tv_in1k",
        "default_name": "resnet18"
    },
    "resnet50": {
        "pretrained_name": "resnet50.tv_in1k",
        "default_name": "resnet50"
    },

    # Modelos optimizados coste/performance
    "efficientnet_b0": {
        "pretrained_name": "efficientnet_b0.ra_in1k",
        "default_name": "efficientnet_b0"
    },
    "efficientnetv2_s": {
        "pretrained_name": "efficientnetv2_s.ra2_in1k",
        "default_name": "efficientnetv2_s"
    },

    # CNN modernos
    "regnety_008": {
        "pretrained_name": "regnety_008.ra_in1k",
        "default_name": "regnety_008"
    },
    "convnext_tiny": {
        "pretrained_name": "convnext_tiny.fb_in1k",
        "default_name": "convnext_tiny"
    },
    "convnext_base": {
        "pretrained_name": "convnext_base.fb_in1k",
        "default_name": "convnext_base"
    },

    # Swin Transformers (jerárquicos) - RECOMENDADO para medical imaging
    "swin_tiny_patch4_window7_224": {
        "pretrained_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "default_name": "swin_tiny_patch4_window7_224"
    },
    "swin_small_patch4_window7_224": {
        "pretrained_name": "swin_small_patch4_window7_224.ms_in1k",
        "default_name": "swin_small_patch4_window7_224"
    },
    "swin_base_patch4_window7_224": {
        "pretrained_name": "swin_base_patch4_window7_224.ms_in1k",
        "default_name": "swin_base_patch4_window7_224"
    },

    # Híbridos CNN + ViT
    "maxvit_tiny_rw_224": {
        "pretrained_name": "maxvit_tiny_rw_224.sw_in1k",
        "default_name": "maxvit_tiny_rw_224"
    },
    "maxvit_base_tf_224": {
        "pretrained_name": "maxvit_base_tf_224.in1k",
        "default_name": "maxvit_base_tf_224"
    },
}


# Diccionario de etiquetas ODIR-5K
ODIR5K_LABELS = {
    'N': {'index': 0, 'name': 'Normal'},
    'D': {'index': 1, 'name': 'Diabetic Retinopathy'},
    'G': {'index': 2, 'name': 'Glaucoma'},
    'C': {'index': 3, 'name': 'Cataract'},
    'A': {'index': 4, 'name': 'Age-related Macular Degeneration'},
    'H': {'index': 5, 'name': 'Hypertension'},
    'M': {'index': 6, 'name': 'Myopia'},
    'O': {'index': 7, 'name': 'Other Diseases/Abnormalities'}
}


# Diccionario de etiquetas EDC
EDC_LABELS = {
    'normal': {'index': 0, 'name': 'Normal'},
    'diabetic_retinopathy': {'index': 1, 'name': 'Diabetic Retinopathy'},
    'glaucoma': {'index': 2, 'name': 'Glaucoma'},
    'cataract': {'index': 3, 'name': 'Cataract'}
}


from codes.data_loader import load_odir5k, load_edc

DATASETS = {
    'edc': {
        'data_folder': './data/eye-diseases-classification',
        'default_img_size': (224, 224),
        'load_func': load_edc,
        'labels_dict': EDC_LABELS,
        'label_format': 'single',
        'image_formats': ('.jpg', '.jpeg'),
    },

    'odir5k': {
        'data_folder': './data/ODIR-5K',
        'default_img_size': (224, 224),
        'load_func': load_odir5k,
        'labels_dict': ODIR5K_LABELS,
        'label_format': 'multi',
        'image_formats': ('.jpg',),
    }
}
