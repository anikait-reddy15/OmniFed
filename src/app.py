import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

app = FastAPI(
    title="OmniFed Control Plane",
    description="Telemetry and management backend for Federated Learning medical imaging.",
    version="1.0.0"
)

class SystemStatus(BaseModel):
    server_address: str
    num_rounds: int
    use_differential_privacy: bool
    status: str

@app.get("/")
def read_root():
    """Returns basic project metadata and status."""
    return {
        "project": "OmniFed",
        "description": "Privacy-Preserving Federated Learning for 3D Medical Imaging",
        "status": "Operational"
    }

@app.get("/api/status", response_model=SystemStatus)
def get_system_status():
    """Returns active configuration and server telemetry parameters."""
    return SystemStatus(
        server_address=config.FL_SERVER_ADDRESS,
        num_rounds=config.NUM_ROUNDS,
        use_differential_privacy=config.USE_DIFFERENTIAL_PRIVACY,
        status="Active"
    )

@app.get("/api/config")
def get_config():
    """Exposes current training hyperparameters for connected nodes."""
    return {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "image_size": config.IMAGE_SIZE,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "noise_multiplier": config.NOISE_MULTIPLIER
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting OmniFed Control Plane on http://{config.SERVER_HOST}:5000")
    uvicorn.run(app, host=config.SERVER_HOST, port=5000)