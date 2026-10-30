import os
import datetime
import ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def generate_mtls_certificates(cert_dir="certs"):
    """
    Generates a Root CA, Server Certificate, and Client Certificate 
    for mTLS communication in OmniFed.
    """
    os.makedirs(cert_dir, exist_ok=True)
    
    print("Generating Root CA private key and certificate...")
    ca_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniFed Medical CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, "OmniFed Root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_private_key, hashes.SHA256())
    )
    
    ca_cert_path = os.path.join(cert_dir, "ca.crt")
    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    print("Generating Server private key and certificate...")
    server_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniFed Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("omnifed-server"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
            ]),
            critical=False,
        )
        .sign(ca_private_key, hashes.SHA256())
    )

    with open(os.path.join(cert_dir, "server.crt"), "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(cert_dir, "server.key"), "wb") as f:
        f.write(server_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    print("Generating Client private key and certificate...")
    client_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniFed Client Node"),
        x509.NameAttribute(NameOID.COMMON_NAME, "omnifed-client-node"),
    ])
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_subject)
        .issuer_name(ca_cert.subject)
        .public_key(client_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .sign(ca_private_key, hashes.SHA256())
    )

    with open(os.path.join(cert_dir, "client.crt"), "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(cert_dir, "client.key"), "wb") as f:
        f.write(client_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    print("All mTLS certificates successfully generated in C:\\OmniFed\\certs")

if __name__ == "__main__":
    generate_mtls_certificates(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs"))