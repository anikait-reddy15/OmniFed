import torch
from opacus import PrivacyEngine
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def apply_differential_privacy(model, optimizer, dataloader):
    """
    Wraps the PyTorch model and optimizer with Opacus PrivacyEngine
    to enable differential privacy during local client training.
    """
    if not config.USE_DIFFERENTIAL_PRIVACY:
        return model, optimizer, None

    privacy_engine = PrivacyEngine()
    
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=config.NOISE_MULTIPLIER,
        max_grad_norm=config.MAX_GRAD_NORM,
    )
    
    return model, optimizer, privacy_engine

if __name__ == "__main__":
    print("Differential privacy utility module loaded successfully.")