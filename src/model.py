import torch
import torch.nn as nn
from monai.networks.nets import UNet

def get_brats_model():
    """
    Initializes and returns a 3D U-Net model configured for BraTS multi-modal MRI segmentation.
    Input channels: 4 (T1, T1Gd, T2, FLAIR)
    Output classes: 3 (Tumor core, whole tumor, enhancing tumor)
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE"
    )
    return model

if __name__ == "__main__":
    # Test model instantiation and forward pass with dummy tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_brats_model().to(device)
    
    # Dummy input matching 8GB VRAM patch size constraint: batch=1, channels=4, spatial=(128, 128, 128)
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    output = net(dummy_input)
    
    print("Model initialized successfully.")
    print("Output tensor shape:", output.shape)