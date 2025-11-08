"""
Module de préparation des données pour la classification de malaria.

Ce module gère le preprocessing des images de cellules sanguines,
incluant le redimensionnement, la normalisation et le split train/test.
"""

import random
from pathlib import Path
from typing import Tuple
from PIL import Image
from tqdm import tqdm

from .config import Config, get_config


def prepare_data(config: Config = None, verbose: bool = True) -> Tuple[int, int]:
    """
    Prépare les données pour l'entraînement.
    
    Redimensionne les images à la taille spécifiée et les sépare
    en ensembles d'entraînement et de test de manière stratifiée.
    
    Args:
        config: Configuration du projet. Si None, charge depuis params.yaml
        verbose: Si True, affiche une barre de progression
        
    Returns:
        Tuple (nb_train_samples, nb_test_samples)
        
    Raises:
        FileNotFoundError: Si le répertoire de données brutes n'existe pas
        
    Example:
        >>> from malaria_classifier import Config, prepare_data
        >>> config = Config.default()
        >>> n_train, n_test = prepare_data(config)
        >>> print(f"Train: {n_train}, Test: {n_test}")
    """
    if config is None:
        config = get_config()
    
    random.seed(config.data.random_seed)
    
    if not config.data.raw_dir.exists():
        raise FileNotFoundError(
            f"Répertoire de données introuvable: {config.data.raw_dir}"
        )
    
    # Création des dossiers de sortie
    for split in ("train", "test"):
        for label in config.model.class_names:
            output_dir = config.data.processed_dir / split / label
            output_dir.mkdir(parents=True, exist_ok=True)
    
    total_train, total_test = 0, 0
    
    # Traitement de chaque classe
    for label in config.model.class_names:
        images = list((config.data.raw_dir / label).glob("*.png"))
        
        if not images:
            print(f"⚠️  Aucune image trouvée pour la classe {label}")
            continue
        
        random.shuffle(images)
        n_test = int(len(images) * config.data.test_split)
        test_imgs = images[:n_test]
        train_imgs = images[n_test:]
        
        # Traitement avec barre de progression
        iterator = [("test", test_imgs), ("train", train_imgs)]
        
        for split, img_list in iterator:
            desc = f"Processing {split}/{label}"
            img_iter = tqdm(img_list, desc=desc, disable=not verbose)
            
            for img_path in img_iter:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(
                    (config.data.img_size, config.data.img_size),
                    Image.Resampling.LANCZOS
                )
                
                dest = config.data.processed_dir / split / label / img_path.name
                img_resized.save(dest, optimize=True)
                
                if split == "train":
                    total_train += 1
                else:
                    total_test += 1
    
    if verbose:
        print(f"\n✓ Préparation terminée:")
        print(f"  - Entraînement: {total_train} images")
        print(f"  - Test: {total_test} images")
    
    return total_train, total_test


def get_dataset_stats(config: Config = None) -> dict:
    """
    Calcule les statistiques du dataset.
    
    Args:
        config: Configuration du projet
        
    Returns:
        Dictionnaire contenant les statistiques (nombres par classe)
    """
    if config is None:
        config = get_config()
    
    stats = {"train": {}, "test": {}}
    
    for split in ("train", "test"):
        for label in config.model.class_names:
            img_dir = config.data.processed_dir / split / label
            if img_dir.exists():
                stats[split][label] = len(list(img_dir.glob("*.png")))
            else:
                stats[split][label] = 0
    
    return stats


if __name__ == "__main__":
    # Point d'entrée pour exécution standalone
    prepare_data(verbose=True)
