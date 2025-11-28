from .get_model import get_model


def get_unlearn_model(name: str, num_classes: int = 10, pretrained: bool = False):
    """
    Alias for clarity: returns a student model to be used for unlearning/KD.
    Separated so you can later diverge (e.g., add heads, masks, etc.).
    """
    return get_model(name, num_classes=num_classes, pretrained=pretrained)