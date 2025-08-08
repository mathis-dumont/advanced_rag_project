import numpy as np
from typing import List
from mistralai import Mistral


class Embedder:
    def __init__(self,
                 api_key: str,
                 model: str = "mistral-embed",
                 batch_size: int = 64):
        """
        :param api_key: clé API pour Mistral
        :param model: nom du modèle d'embedding
        :param batch_size: taille des lots pour les appels API
        """
        # client Mistral pour les embeddings
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.

        :param texts: liste de chaînes à encoder
        :return: tableau numpy de dimension (len(texts), dim_embedding)
        """
        vectors: List[List[float]] = []
        # traitement par paquets pour optimiser les appels API
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                inputs=batch
            ).data
            # extraire les embeddings
            for d in response:
                vectors.append(d.embedding)
        # conversion en numpy array
        return np.array(vectors, dtype="float32")
