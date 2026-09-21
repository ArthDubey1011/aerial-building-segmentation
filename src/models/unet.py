"""Model builder: U-Net with an ImageNet-pretrained ResNet34 encoder (segmentation-models-pytorch)."""
import segmentation_models_pytorch as smp


def build_model(cfg):
    """Returns a U-Net that outputs raw logits of shape (B, 1, H, W) (no sigmoid: losses apply it themselves,
    which is numerically more stable, e.g. BCEWithLogits)."""
    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"],  # "imagenet" downloads weights once; None = random init
        in_channels=3,
        classes=m["classes"],
        activation=None,
    )
