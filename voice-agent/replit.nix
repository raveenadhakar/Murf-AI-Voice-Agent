run = "uvicorn main:app --host=0.0.0.0 --port=8000"

modules = ["python-3.11"]

[nix]
packages = ["libxcrypt"]

[ports]
localPort = 8000
externalPort = 80