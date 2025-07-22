import os
import sys
from pathlib import Path

# Test simple
print("Python fonctionne!")
print(f"Version: {sys.version}")
print(f"Répertoire: {os.getcwd()}")

# Test Pillow
try:
    from PIL import Image
    print("Pillow OK")
except ImportError:
    print("Pillow manquant")

# Test yaml
try:
    import yaml
    print("YAML OK")
except ImportError:
    print("YAML manquant")

# Test structure
if Path("data/processed/train/Parasitized").exists():
    count = len(list(Path("data/processed/train/Parasitized").glob("*.png")))
    print(f"Train Parasitized: {count} images")

if Path("data/processed/test/Parasitized").exists():
    count = len(list(Path("data/processed/test/Parasitized").glob("*.png")))
    print(f"Test Parasitized: {count} images")
