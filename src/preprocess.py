import os
import yaml
import random
from PIL import Image
from pathlib import Path

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def resize_and_split(raw_dir, out_dir, img_size, test_split, seed):
    random.seed(seed)

    # Création des dossiers train/test
    for split in ("train", "test"):
        for label in ("Parasitized", "Uninfected"):
            make_dirs(Path(out_dir) / split / label)

    # Parcours des classes
    for label in ("Parasitized", "Uninfected"):
        imgs = list(Path(raw_dir, label).glob("*.png"))
        random.shuffle(imgs)
        n_test = int(len(imgs) * test_split)
        test_imgs = imgs[:n_test]
        train_imgs = imgs[n_test:]

        for split, group in (("test", test_imgs), ("train", train_imgs)):
            for img_path in group:
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))
                dest = Path(out_dir) / split / label / img_path.name
                img.save(dest)

def main():
    params = load_params()
    raw = params["data"]["raw_dir"]
    out  = params["data"]["processed_dir"]
    p    = params["preprocess"]

    resize_and_split(
        raw_dir=raw,
        out_dir=out,
        img_size=p["img_size"],
        test_split=p["test_split"],
        seed=p["random_seed"],
    )

if __name__ == "__main__":
    main()
