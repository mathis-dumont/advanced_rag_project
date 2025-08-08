# src/app.py

import argparse
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from rag_agent.config import load_settings
from rag_agent.pipeline import RAGPipeline

# Configuration du logging pour l'application CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Point d'entrée principal pour l'interface en ligne de commande."""
    
    # Charger les variables d'environnement (ex: MISTRAL_API_KEY) depuis un fichier .env
    load_dotenv()
    
    # 1. Charger la configuration depuis settings.json
    try:
        settings = load_settings(Path("settings.json"))
    except FileNotFoundError as e:
        logging.error("ERREUR CRITIQUE: %s. Veuillez créer ce fichier.", e)
        return

    # 2. Gérer les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Système de RAG basé sur des documents.")
    parser.add_argument(
        "--mode",
        choices=["auto", "rebuild", "incremental"],
        default="auto",
        help="Mode de mise à jour de la base (auto, rebuild, incremental)."
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Convertir les documents Word en PDF sans construire l'index."
    )
    parser.add_argument("--query", type=str, help="Question à poser directement.")
    parser.add_argument(
        "--process-images",
        action=argparse.BooleanOptionalAction,
        default=settings.process_images,
        help="Activer/Désactiver le traitement des images par l'IA."
    )
    args = parser.parse_args()

    # Mettre à jour les settings avec les arguments de la CLI
    settings.process_images = args.process_images
    
    # 3. Initialiser le pipeline (logique centralisée et propre)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logging.error("ERREUR CRITIQUE: La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
        return
        
    try:
        pipeline = RAGPipeline(settings=settings, api_key=api_key)
    except Exception as e:
        logging.error("Erreur fatale lors de l'initialisation du pipeline: %s", e, exc_info=True)
        return

    # 4. Exécuter la logique de l'application
    if args.convert_only:
        logging.info("Mode 'convert-only' activé.")
        # La logique de conversion est maintenant gérée par le loader, mais on peut la forcer ici
        for doc in pipeline.loader.data_dir.rglob("*.[dD][oO][cC]*"):
            try:
                pdf = pipeline.converter.to_pdf(doc)
                logging.info(f"✅ Converti : {doc.name} → {pdf.name}")
            except Exception as e:
                logging.error(f"⚠️ Échec conversion {doc.name}: {e}")
        return

    if args.query:
        logging.info("Mode 'query' : réponse à une question unique.")
        réponse = pipeline.answer(args.query, update_mode=args.mode)
        print("\n--- Réponse ---\n")
        print(réponse)
        return

    # Mode interactif par défaut
    if args.mode != "auto":
        pipeline.build_or_update(mode=args.mode)

    print("\n--- Assistant Technique Interactif ---")
    print("Tapez 'q' ou 'exit' pour quitter.")
    while True:
        try:
            q = input("\nQuestion › ")
            if q.lower() in ("q", "exit", "quit"):
                break
            réponse = pipeline.answer(q)
            print("\n--- Réponse ---\n")
            print(réponse)
        except (KeyboardInterrupt, EOFError):
            break
    print("\nSession terminée.")

if __name__ == "__main__":
    main()