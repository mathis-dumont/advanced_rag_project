import faiss
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .chunking import Chunk


class IndexManager:
    def __init__(self,
                 db_dir: Path,
                 dim: int = 1024):
        """
        Gestionnaire de l'index FAISS et des chunks associés.

        :param db_dir: répertoire pour stocker index et chunks
        :param dim: dimension des embeddings (pour création d'un index vide)
        """
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "faiss_index.idx"
        self.chunks_path = self.db_dir / "chunks.pkl"
        self.dim = dim

    def build(self, vectors: np.ndarray) -> faiss.Index:
        """
        Construit un index FAISS à partir des vecteurs donnés.

        :param vectors: array shape (n_samples, dim)
        :return: index FAISS (IP)
        """
        # Normalisation L2 pour similarité cosinus
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def save(self,
             index: faiss.Index,
             chunks: List[Chunk]) -> None:
        """
        Sauvegarde l'index FAISS et la liste de chunks.

        :param index: index FAISS
        :param chunks: liste des Chunk correspondants aux vecteurs
        """
        # Écriture de l'index
        faiss.write_index(index, str(self.index_path))
        # Sérialisation des chunks
        with open(self.chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"💾 Base sauvegardée ({len(chunks)} chunks au total)")

    def load(self) -> Tuple[faiss.Index, List[Chunk]]:
        """
        Charge l'index FAISS et la liste de chunks depuis le disque.

        :return: (index, chunks)
        :raises FileNotFoundError: si index ou chunks manquent
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Index ou file de chunks manquant dans {self.db_dir}"
            )
        # Lecture de l'index
        index = faiss.read_index(str(self.index_path))
        # Désérialisation des chunks
        with open(self.chunks_path, "rb") as f:
            chunks: List[Chunk] = pickle.load(f)
        return index, chunks
