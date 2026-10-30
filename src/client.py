import os
import sys
from pathlib import Path
import torch
import flwr as fl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from model import get_brats_model
from dataset import get_dataloader

class BraTSClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, device):
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.LEARNING_RATE
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v, device=self.device) for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        use_cuda = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_cuda else None
        
        for epoch in range(cfg.LOCAL_EPOCHS):
            for batch in self.trainloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                self.optimizer.zero_grad()
                
                if use_cuda:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        if labels.ndim == 5:
                            labels = torch.argmax(labels, dim=1)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    if labels.ndim == 5:
                        labels = torch.argmax(labels, dim=1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                if cfg.USE_DIFFERENTIAL_PRIVACY and use_cuda:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.MAX_GRAD_NORM
                    )

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.trainloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                if labels.ndim == 5:
                    labels = torch.argmax(labels, dim=1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                
        loss = total_loss / len(self.trainloader.dataset)
        return float(loss), len(self.trainloader.dataset), {"loss": float(loss)}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing client on device: {device}")
    
    model = get_brats_model().to(device)
    trainloader = get_dataloader()
    
    client = BraTSClient(model, trainloader, device)
    print(f"Connecting to Flower server at {cfg.FL_SERVER_ADDRESS}...")
    
    root_certificates = None
    if cfg.ENABLE_MTLS:
        cert_dir = cfg.CERT_DIR
        root_certificates = Path(os.path.join(cert_dir, "ca.crt")).read_bytes()
        print("mTLS root certificates loaded for secure client connection.")

    fl.client.start_client(
        server_address=cfg.FL_SERVER_ADDRESS,
        client=client.to_client(),
        root_certificates=root_certificates
    )