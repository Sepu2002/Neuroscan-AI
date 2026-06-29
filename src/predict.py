"""
NeuroScan AI — standalone inference
------------------------------------
Load a trained model checkpoint and run prediction + Grad-CAM on any MRI image.

Usage:
    from src.predict import load_model, predict

    model = load_model("neuroscan_ai_final.pt")
    result = predict("path/to/scan.jpg", model, show_gradcam=True)
    print(result)
"""

import torch
import torchvision.transforms as T
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DESCRIPCIONES = {
    "glioma":     "Tumor originado en células gliales. Requiere atención urgente.",
    "meningioma": "Tumor en las meninges, generalmente de crecimiento lento.",
    "notumor":    "No se detectaron indicios de tumor en la imagen.",
    "pituitary":  "Tumor en la glándula pituitaria. Tratable con cirugía/radioterapia.",
}


def load_model(checkpoint_path: str, device: str = None):
    """Load a saved NeuroScan AI checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict(image_path: str, model=None, class_names=None, device=None,
            checkpoint_path: str = "neuroscan_ai_final.pt",
            show_gradcam: bool = True) -> dict:
    """
    Classify a brain MRI scan and optionally display a Grad-CAM overlay.

    Args:
        image_path     : Path to the MRI image file.
        model          : Pre-loaded model (optional — loads from checkpoint if None).
        class_names    : List of class names (optional — read from checkpoint if None).
        device         : 'cuda' or 'cpu' (optional — auto-detected if None).
        checkpoint_path: Path to the .pt checkpoint file.
        show_gradcam   : If True, display MRI + heatmap + probability chart.

    Returns:
        dict with keys: clase, confianza, descripcion, probabilidades
    """
    if model is None:
        model, class_names, device = load_model(checkpoint_path)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_pil    = Image.open(image_path).convert("RGB")
    img_224    = img_pil.resize((224, 224))
    img_np     = np.array(img_224) / 255.0
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits     = model(input_tensor)
        probs      = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        pred_clase = class_names[pred_idx]
        pred_conf  = probs[pred_idx].item()

    resultado = {
        "clase":      pred_clase,
        "confianza":  round(pred_conf * 100, 2),
        "descripcion": DESCRIPCIONES.get(pred_clase, ""),
        "probabilidades": {
            class_names[i]: round(probs[i].item() * 100, 2)
            for i in range(len(class_names))
        },
    }

    if show_gradcam:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            target_layer = [model.features[-1]]
            with GradCAM(model=model, target_layers=target_layer) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(pred_idx)]
                )[0]
            visualization = show_cam_on_image(
                img_np.astype(np.float32), grayscale_cam, use_rgb=True
            )

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig.suptitle(
                f"NeuroScan AI — {pred_clase.upper()} ({pred_conf*100:.1f}%)",
                fontsize=13, fontweight="bold"
            )

            axes[0].imshow(img_np)
            axes[0].set_title("MRI Original")
            axes[0].axis("off")

            axes[1].imshow(visualization)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            clases     = list(resultado["probabilidades"].keys())
            probs_vals = list(resultado["probabilidades"].values())
            colores    = ["#E24B4A" if c == pred_clase else "#A0AEC0" for c in clases]
            axes[2].barh(clases, probs_vals, color=colores)
            axes[2].set_xlim(0, 100)
            axes[2].set_xlabel("Probability (%)")
            axes[2].set_title("Confidence by class")
            axes[2].spines["top"].set_visible(False)
            axes[2].spines["right"].set_visible(False)
            for i, v in enumerate(probs_vals):
                axes[2].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Grad-CAM not available. Install with: pip install grad-cam")

    return resultado
