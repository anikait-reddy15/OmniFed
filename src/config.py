import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CERT_DIR = os.path.join(BASE_DIR, "certs")

# Dynamic server host: reads environment variable if running in Docker, else defaults to localhost
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = 8080
FL_SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"

# Training Hyperparameters
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
IMAGE_SIZE = (128, 128, 128)

# Security Settings
USE_DIFFERENTIAL_PRIVACY = True
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 0.1
ENABLE_MTLS = True