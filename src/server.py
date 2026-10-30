import os
import sys
from pathlib import Path
import flwr as fl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    print(f"Starting OmniFed FL Server on {config.FL_SERVER_ADDRESS}...")

    certificates = None
    if config.ENABLE_MTLS:
        cert_dir = config.CERT_DIR
        certificates = (
            Path(os.path.join(cert_dir, "ca.crt")).read_bytes(),
            Path(os.path.join(cert_dir, "server.crt")).read_bytes(),
            Path(os.path.join(cert_dir, "server.key")).read_bytes(),
        )
        print("mTLS encryption and mutual authentication enabled on server.")

    fl.server.start_server(
        server_address=config.FL_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
        certificates=certificates,
    )

if __name__ == "__main__":
    main()