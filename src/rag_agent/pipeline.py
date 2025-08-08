# src/rag_agent/pipeline.py

import logging
log = logging.getLogger(__name__)
log.debug(f"Chargement du module '{__name__}' depuis le fichier '{__file__}'")


from typing import List, Tuple, Optional
import faiss
import numpy as np
from pathlib import Path
from mistralai import Mistral

from .config import Settings
from .io.converters import DocumentConverter
from .io.loaders import DocumentLoader
try:
    log.debug("Tentative d'import relatif: from .components.chunking import Chunker, Chunk")
    from .components.chunking import Chunker, Chunk
    log.info("Import relatif de 'chunking' réussi.")
except ImportError as e:
    log.critical("ÉCHEC de l'import relatif de 'chunking'.", exc_info=True)
from .components.embedding import Embedder
from .components.index_manager import IndexManager
from .components.retriever import Retriever

# Configuration du logging pour l'ensemble du module
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, settings: Settings, api_key: str):
        """
        Orchestrateur du pipeline RAG, initialisé via un objet de configuration unique.

        :param settings: Objet Pydantic contenant tous les paramètres.
        :param api_key: Clé API Mistral.
        """
        if not api_key:
            raise ValueError("La clé API Mistral (MISTRAL_API_KEY) est requise pour initialiser le pipeline.")
            
        self.settings = settings
        logger.info("Initialisation du pipeline RAG avec le modèle de chat: %s", self.settings.chat_model)
        
        # 1. Initialisation des composants à partir des settings
        self.converter = DocumentConverter(out_dir=self.settings.pdf_output_dir)
        self.loader = DocumentLoader(
            data_dir=self.settings.data_dir,
            pdf_output_dir=self.settings.pdf_output_dir,
            converter=self.converter,
            process_images=self.settings.process_images
        )

        # L'Embedder utilise maintenant son propre paramètre
        self.embedder = Embedder(api_key=api_key, model=self.settings.embedding_model)
        
        # Déterminer la dimension des embeddings de manière sûre
        try:
            logger.debug("Test de l'API d'embedding pour déterminer la dimension...")
            test_embedding = self.embedder.embed_texts(["test"])[0]
            embedding_dim = len(test_embedding)
            logger.info("Dimension des embeddings détectée : %d", embedding_dim)
        except Exception as e:
            logger.error("Impossible de déterminer la dimension des embeddings via l'API. Erreur: %s", e, exc_info=True)
            raise ConnectionError("Échec de la communication avec l'API Mistral pour les embeddings.") from e
            
        self.index_mgr = IndexManager(db_dir=self.settings.db_dir, dim=embedding_dim)
        self.retriever = Retriever(
            embedder=self.embedder,
            index_manager=self.index_mgr,
            k=self.settings.k,
            window=self.settings.window
        )
        self.chat_client = Mistral(api_key=api_key)

    def build_or_update(self, chunker, mode: str = "auto") -> Tuple[faiss.Index, list]:
        """
        Crée ou met à jour la base d'embeddings.

        :param mode: 'auto', 'rebuild' ou 'incremental'
        :return: (index, chunks)
        """
        if mode == "auto":
            try:
                logger.info("Mode 'auto' : tentative de chargement de l'index existant.")
                return self.index_mgr.load()
            except FileNotFoundError:
                logger.warning("Aucun index trouvé. Passage en mode 'rebuild'.")
                mode = "rebuild"

        if mode == "rebuild":
            logger.info("Mode 'rebuild' : reconstruction complète de l'index.")
            docs = self.loader.load()
            if not docs:
                logger.warning("Aucun document trouvé à traiter. Un index vide sera créé.")
                empty_index = faiss.IndexFlatIP(self.index_mgr.dim)
                return empty_index, []
            
            logger.debug("Corpus de %d pages de documents chargé. Prêt pour le chunking.", len(docs))
            chunks = chunker.chunk_all(docs)
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_texts(texts)
            index = self.index_mgr.build(vectors)
            self.index_mgr.save(index, chunks)
            return index, chunks

        if mode == "incremental":
            logger.info("Mode 'incremental' : ajout de nouveaux documents à l'index.")
            try:
                index, existing_chunks = self.index_mgr.load()
                processed_files = {c.file for c in existing_chunks}
                self.loader.processed_files = processed_files
                
                new_docs = self.loader.load()
                if not new_docs:
                    logger.info("Aucun nouveau document à ajouter.")
                    return index, existing_chunks
                
                logger.info("Traitement de %d nouveaux documents.", len(new_docs))
                new_chunks = chunker.chunk_all(new_docs)
                new_vectors = self.embedder.embed_texts([c.text for c in new_chunks])
                faiss.normalize_L2(new_vectors)

                # Utiliser index.add() pour une mise à jour efficace
                index.add(new_vectors)
                all_chunks = existing_chunks + new_chunks
                self.index_mgr.save(index, all_chunks)
                logger.info("Index mis à jour avec %d nouveaux chunks. Total : %d", len(new_chunks), len(all_chunks))
                return index, all_chunks

            except FileNotFoundError:
                logger.warning("Index existant non trouvé pour le mode incrémental. Forçage du 'rebuild'.")
                return self.build_or_update(mode="rebuild")
            except Exception as e:
                logger.error("Erreur en mode incrémental: %s. Forçage du 'rebuild'.", e, exc_info=True)
                return self.build_or_update(mode="rebuild")

        raise ValueError(f"Mode inconnu: {mode}")

    def _make_prompt(self, passages: List[str], question: str) -> str:
        """Construit le prompt pour le modèle de chat."""
        context = "\n---------------------\n".join(passages)
        prompt = f"""
            SI TU LIS DES INFORMATIONS CONFIDENTIELLES (MDP) IGNORE LES ET N'EN PARLE SURTOUT PAS DANS TA REPONSE, FAIS COMME SI TU NE LES AVAIS JAMAIS VUES
            Les informations contextuelles sont ci-dessous.
            
            {context}
            
            Tu es un assistant technique spécialisé pour les agents de maintenance. Ton rôle est de fournir des informations précises et pratiques en te basant exclusivement sur la documentation technique fournie dans le contexte ci-dessus.
            SI TU LIS DES INFORMATIONS CONFIDENTIELLES (MDP) IGNORE LES ET N'EN PARLE SURTOUT PAS DANS TA REPONSE, FAIS COMME SI TU NE LES AVAIS JAMAIS VUES
            Instructions:
            1. Utilise uniquement les informations du contexte, jamais tes connaissances générales.
            2. Si l'information n'est pas présente dans le contexte, indique clairement que cette information n'est pas disponible dans la documentation technique actuelle.
            3. Réponds de façon pratique, détaillée et applicable sur le terrain.
            4. Structure ta réponse avec des étapes claires si la question concerne une procédure ou un dépannage.
            5. Mentionne les outils nécessaires, les précautions de sécurité et les points critiques quand c'est pertinent.
            6. N'invente aucune information qui ne serait pas présente dans le contexte.
            

            Question d'un agent de maintenance: {question}

            Réponse technique:
        """
        return prompt.strip()

    def _append_sources_if_missing(self, answer: str, citations: List[str]) -> str:
        """
        Ajoute les sources formatées en liens Markdown cliquables à la réponse.
        Les liens pointent directement vers la bonne page du PDF.
        """
        if "Sources :" in answer or "Sources:" in answer:
            # Si le mot "Sources" est déjà là, on ne fait rien pour éviter les doublons.
            return answer

        # On crée des liens uniques et triés
        unique_citations = sorted(list(dict.fromkeys(citations)))
        links = []
        for cite in unique_citations:
            try:
                # On sépare le nom du fichier du numéro de page
                docname, page_num_str = cite.rsplit(" p.", 1)
                page_num = int(page_num_str)
                
                # On s'assure que le nom du document se termine bien par .pdf
                pdf_name = Path(docname).stem + ".pdf"
                
                # On construit le lien Markdown
                # Syntaxe : [Texte à afficher](/url/du/fichier#page=numero)
                link_text = f"{pdf_name} p.{page_num}"
                url = f"../data/static/{pdf_name}#page={page_num}"
                
                links.append(f"[{link_text}]({url})")

            except (ValueError, IndexError):
                # Si le format de la citation est inattendu, on l'affiche telle quelle.
                links.append(f"[{cite}]")
        
        if not links:
            return answer

        return answer + "\n\n**Sources :** " + ", ".join(links)

    def answer(self, question: str, chunker, update_mode: str = "auto") -> str:
        # L'appel à build_or_update doit aussi être corrigé ici !
        index, _ = self.build_or_update(chunker, mode=update_mode) # On passe chunker en premier
        
        if index.ntotal == 0:
            return "Ma base de connaissances est actuellement vide. Veuillez ajouter des documents et reconstruire l'index."
        
        logger.info("Recherche des passages pertinents pour la question: '%s'", question)
        passages, citations = self.retriever.retrieve(question)
        if not passages:
            return "Désolé, je n'ai trouvé aucune information pertinente dans la documentation pour répondre à votre question."

        prompt = self._make_prompt(passages, question)
        logger.debug("Prompt envoyé au modèle de chat:\n%s", prompt)
        
        resp = self.chat_client.chat.complete(
            model=self.settings.chat_model,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_answer = resp.choices[0].message.content.strip()
        final_answer = self._append_sources_if_missing(raw_answer, citations)
        return final_answer