import os
import glob
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    EnsureTyped,
)
from monai.data import Dataset
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_brats_file_list(data_dir=config.DATA_DIR):
    """
    Scans the BraTS training directory for real patient folders and builds file path dictionaries.
    Ignores @PaxHeader archive metadata.
    """
    training_data_path = os.path.join(data_dir, "BraTS2021_Training_Data")
    if not os.path.exists(training_data_path):
        training_data_path = data_dir

    if not os.path.exists(training_data_path):
        return []

    patient_folders = [
        os.path.join(training_data_path, d) for d in os.listdir(training_data_path)
        if os.path.isdir(os.path.join(training_data_path, d)) and not d.startswith("@")
    ]
    
    data_dicts = []
    for p_dir in patient_folders:
        base_name = os.path.basename(p_dir)
        if base_name.startswith("@"):
            continue
            
        flair = os.path.join(p_dir, f"{base_name}_flair.nii.gz")
        t1 = os.path.join(p_dir, f"{base_name}_t1.nii.gz")
        t1ce = os.path.join(p_dir, f"{base_name}_t1ce.nii.gz")
        t2 = os.path.join(p_dir, f"{base_name}_t2.nii.gz")
        seg = os.path.join(p_dir, f"{base_name}_seg.nii.gz")
        
        # Verify all required modalities exist for this patient
        if os.path.exists(flair) and os.path.exists(t1) and os.path.exists(t1ce) and os.path.exists(t2) and os.path.exists(seg):
            data_dicts.append({
                "image": [flair, t1, t1ce, t2],
                "label": seg
            })
            
    return data_dicts

def get_dataloader(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    """
    Returns a PyTorch DataLoader for training clients using real BraTS .gz files if available,
    otherwise falls back to synthetic tensors.
    """
    data_dicts = get_brats_file_list(data_dir)
    
    if len(data_dicts) > 0:
        print(f"Loaded {len(data_dicts)} real patient volumes from BraTS dataset.")
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=config.IMAGE_SIZE,
                    random_size=False,
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        dataset = Dataset(data=data_dicts, transform=transforms)
    else:
        print("No real BraTS patient folders found in data directory. Using synthetic tensor fallback.")
        os.makedirs(data_dir, exist_ok=True)
        samples = []
        for i in range(4):
            patient_id = f"BraTS2021_{i:05d}"
            img_path = os.path.join(data_dir, f"{patient_id}_image.pt")
            lbl_path = os.path.join(data_dir, f"{patient_id}_label.pt")
            if not os.path.exists(img_path):
                torch.save(torch.randn(4, 128, 128, 128), img_path)
                torch.save(torch.randint(0, 2, (3, 128, 128, 128)), lbl_path)
            samples.append({"image": img_path, "label": lbl_path})
            
        class SyntheticBraTSDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __len__(self):
                return len(self.data_list)
            def __getitem__(self, idx):
                item = self.data_list[idx]
                return {"image": torch.load(item["image"]), "label": torch.load(item["label"])}
                
        dataset = SyntheticBraTSDataset(samples)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    return dataloader

if __name__ == "__main__":
    loader = get_dataloader()
    for batch in loader:
        print("Data loader test successful.")
        print("Image Tensor shape:", batch["image"].shape)
        print("Label Tensor shape:", batch["label"].shape)
        break