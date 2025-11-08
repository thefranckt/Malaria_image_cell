#!/usr/bin/env python
"""
Interface en ligne de commande pour le projet Malaria Cell Classification.

Ce module fournit un point d'entrée unique pour toutes les opérations:
préparation des données, entraînement, prédiction et déploiement.
"""

import argparse
import sys
from pathlib import Path

from src.config import Config, get_config
from src.data import prepare_data, get_dataset_stats
from src.model import train_model
from src.inference import MalariaClassifier


def cmd_prepare(args):
    """Commande: Préparer les données."""
    config = get_config()
    print("🔄 Préparation des données...")
    n_train, n_test = prepare_data(config, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\n✅ Données préparées avec succès!")


def cmd_train(args):
    """Commande: Entraîner le modèle."""
    config = get_config()
    
    # Override des paramètres si fournis
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    
    print("🚀 Démarrage de l'entraînement...")
    model, history = train_model(config, verbose=not args.quiet)


def cmd_predict(args):
    """Commande: Faire une prédiction."""
    classifier = MalariaClassifier(model_path=args.model)
    
    if args.batch:
        # Prédiction batch
        image_paths = [Path(p) for p in args.image]
        results = classifier.predict_batch(image_paths)
        
        for result in results:
            if result['success']:
                print(f"\n📸 {result['image']}")
                print(f"   Classe: {result['class']}")
                print(f"   Confiance: {result['confidence']:.2%}")
            else:
                print(f"\n❌ {result['image']}: {result['error']}")
    else:
        # Prédiction simple
        result = classifier.predict(args.image[0])
        
        print(f"\n🔬 Résultat de l'analyse:")
        print(f"   Image: {args.image[0]}")
        print(f"   Classe: {result['class']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        
        if args.verbose:
            print(f"\n📊 Probabilités détaillées:")
            for cls, prob in result['probabilities'].items():
                bar = "█" * int(prob * 50)
                print(f"   {cls:15s}: {prob:.4f} {bar}")


def cmd_stats(args):
    """Commande: Afficher les statistiques du dataset."""
    config = get_config()
    stats = get_dataset_stats(config)
    
    print("\n📊 Statistiques du dataset:")
    print("\n🎯 Entraînement:")
    for cls, count in stats['train'].items():
        print(f"   {cls:15s}: {count:6d} images")
    print(f"   {'Total':15s}: {sum(stats['train'].values()):6d} images")
    
    print("\n🎯 Test:")
    for cls, count in stats['test'].items():
        print(f"   {cls:15s}: {count:6d} images")
    print(f"   {'Total':15s}: {sum(stats['test'].values()):6d} images")
    
    total = sum(stats['train'].values()) + sum(stats['test'].values())
    print(f"\n📈 Total général: {total} images")


def cmd_serve(args):
    """Commande: Lancer le serveur API."""
    try:
        from api import app
        print(f"🌐 Démarrage du serveur sur http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    except ImportError:
        print("❌ Flask n'est pas installé. Installez-le avec: pip install flask")
        sys.exit(1)


def main():
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description="Malaria Cell Classification - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Préparer les données
  python cli.py prepare
  
  # Entraîner le modèle
  python cli.py train --epochs 20 --batch-size 64
  
  # Faire une prédiction
  python cli.py predict image.png
  
  # Prédiction batch
  python cli.py predict img1.png img2.png --batch
  
  # Lancer le serveur API
  python cli.py serve --port 8080
  
  # Afficher les statistiques
  python cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande: prepare
    parser_prepare = subparsers.add_parser(
        'prepare',
        help='Préparer les données (redimensionnement et split)'
    )
    parser_prepare.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Mode silencieux (pas de barre de progression)'
    )
    
    # Commande: train
    parser_train = subparsers.add_parser(
        'train',
        help='Entraîner le modèle'
    )
    parser_train.add_argument(
        '-e', '--epochs',
        type=int,
        help='Nombre d\'époques'
    )
    parser_train.add_argument(
        '-b', '--batch-size',
        type=int,
        help='Taille du batch'
    )
    parser_train.add_argument(
        '--lr',
        type=float,
        help='Learning rate'
    )
    parser_train.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Mode silencieux'
    )
    
    # Commande: predict
    parser_predict = subparsers.add_parser(
        'predict',
        help='Faire une prédiction sur une ou plusieurs images'
    )
    parser_predict.add_argument(
        'image',
        nargs='+',
        help='Chemin(s) vers l\'image ou les images'
    )
    parser_predict.add_argument(
        '-m', '--model',
        default='models/model.pth',
        help='Chemin vers le modèle (défaut: models/model.pth)'
    )
    parser_predict.add_argument(
        '--batch',
        action='store_true',
        help='Mode batch pour plusieurs images'
    )
    parser_predict.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Afficher les probabilités détaillées'
    )
    
    # Commande: stats
    parser_stats = subparsers.add_parser(
        'stats',
        help='Afficher les statistiques du dataset'
    )
    
    # Commande: serve
    parser_serve = subparsers.add_parser(
        'serve',
        help='Lancer le serveur API Flask'
    )
    parser_serve.add_argument(
        '--host',
        default='0.0.0.0',
        help='Adresse d\'écoute (défaut: 0.0.0.0)'
    )
    parser_serve.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port d\'écoute (défaut: 5000)'
    )
    parser_serve.add_argument(
        '--debug',
        action='store_true',
        help='Mode debug'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch des commandes
    commands = {
        'prepare': cmd_prepare,
        'train': cmd_train,
        'predict': cmd_predict,
        'stats': cmd_stats,
        'serve': cmd_serve,
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Opération interrompue par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        if args.command in ['train', 'prepare'] and hasattr(args, 'quiet') and not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
