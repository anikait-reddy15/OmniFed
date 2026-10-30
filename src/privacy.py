import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from typing import Tuple

def setup_differential_privacy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    target_delta: float = 1e-5
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    """
    Wraps the PyTorch/MONAI model, optimizer, and data loader with Opacus 
    to enforce Differential Privacy via per-sample gradient clipping and noise injection.
    """
    privacy_engine = PrivacyEngine()
    
    # Attach model, optimizer, and dataloader to Opacus PrivacyEngine
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    
    print("================================================================================")
    print("                    DIFFERENTIAL PRIVACY ENGINE INITIALIZED                     ")
    print("================================================================================")
    print(f"Target Noise Multiplier (sigma) : {noise_multiplier}")
    print(f"Max Gradient Clipping Norm (C)  : {max_grad_norm}")
    print(f"Target Delta (delta)            : {target_delta}")
    print("================================================================================")
    
    return model, optimizer, data_loader, privacy_engine

def compute_privacy_budget(privacy_engine: PrivacyEngine, target_delta: float = 1e-5) -> float:
    """
    Calculates and prints the accumulated privacy expenditure (epsilon) for a given delta.
    """
    epsilon = privacy_engine.get_epsilon(target_delta=target_delta)
    print(f"[PRIVACY BUDGET] Current Epsilon (epsilon): {epsilon:.4f} | Delta (delta): {target_delta}")
    return epsilon

if __name__ == "__main__":
    # Offline Verification Test Script
    print("\nRunning Offline Differential Privacy Utility Test...\n")
    
    # 1. Dummy Model (Simulating lightweight layer / model)
    dummy_model = nn.Sequential(
        nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(16, 3)
    )
    
    # 2. Dummy Optimizer and Data
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
    
    # Create dummy batch (Batch size: 4, Channels: 4, Depth: 32, Height: 32, Width: 32)
    dummy_inputs = torch.randn(4, 4, 32, 32, 32)
    dummy_targets = torch.randn(4, 3)
    
    dataset = torch.utils.data.TensorDataset(dummy_inputs, dummy_targets)
    dummy_loader = DataLoader(dataset, batch_size=2)
    
    # 3. Attach Differential Privacy
    p_model, p_optimizer, p_loader, p_engine = setup_differential_privacy(
        model=dummy_model,
        optimizer=dummy_optimizer,
        data_loader=dummy_loader,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        target_delta=1e-5
    )
    
    # 4. Simulate Local Training Steps
    p_model.train()
    for epoch in range(1, 3):
        for step, (x, y) in enumerate(p_loader):
            p_optimizer.zero_grad()
            outputs = p_model(x)
            loss = nn.functional.mse_loss(outputs, y)
            loss.backward()
            p_optimizer.step()
            
        eps = compute_privacy_budget(p_engine, target_delta=1e-5)
        
    print("\n[SUCCESS] Privacy utility verification complete.")