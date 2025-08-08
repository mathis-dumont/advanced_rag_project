from typing import List, Tuple, Optional
import faiss

from .chunking import Chunk
from .embedding import Embedder
from .index_manager import IndexManager


class Retriever:
    def __init__(self,
                 embedder: Embedder,
                 index_manager: IndexManager,
                 k: int = 10,
                 window: int = 1):
        """
        :param embedder: instance d'Embedder pour transformer les requêtes en vecteurs
        :param index_manager: instance d'IndexManager pour charger l'index et les chunks
        :param k: nombre de voisins à rechercher
        :param window: nombre de chunks supplémentaires de chaque côté pour le contexte
        """
        self.embedder = embedder
        self.index_manager = index_manager
        self.k = k
        self.window = window

    def retrieve(self,
                 question: str,
                 k: Optional[int] = None,
                 window: Optional[int] = None
                 ) -> Tuple[List[str], List[str]]:
        """
        Recherche les passages les plus pertinents pour la question donnée.

        :param question: chaîne de caractères de la question à poser
        :param k: (optionnel) nombre de voisins à récupérer (défaut: self.k)
        :param window: (optionnel) taille du contexte adjacent (défaut: self.window)
        :return: tuple (passages, citations)
        """
        # paramètres effectifs
        k = k if k is not None else self.k
        window = window if window is not None else self.window

        # Chargement de l'index et des chunks
        index, chunks = self.index_manager.load()

        # Encodage de la question
        q_vec = self.embedder.embed_texts([question])
        faiss.normalize_L2(q_vec)

        # Recherche k plus proches voisins
        distances, indices = index.search(q_vec, k)

        passages: List[str] = []
        citations: List[str] = []
        for idx in indices[0]:
            # définir la plage de contexte
            start = max(0, idx - window)
            end = idx + window + 1
            # concaténer les textes du contexte
            context_chunks = chunks[start:end]
            passages.append("\n".join(c.text for c in context_chunks))
            # citation du chunk pivot
            pivot = chunks[idx]
            citations.append(f"{pivot.file} p.{pivot.page}")

        return passages, citations
