# Dans src/rag_agent/components/chunking.py

from dataclasses import dataclass
from typing import List, Tuple
import tiktoken
# NOTE : L'import de spacy est toujours supprimé d'ici

@dataclass
class Chunk:
    """
    Représente un chunck de texte extrait d'une page.
    Attributes:
        text: contenu textuel du chunk
        file: nom du fichier source
        page: numéro de page dans le PDF
    """
    text: str
    file: str
    page: int


class Chunker:
    def __init__(self,
                 nlp_model: str = "fr_core_news_md",
                 token_model: str = "cl100k_base",
                 max_tokens: int = 350,
                 overlap: int = 40):
        """
        :param nlp_model: nom du modèle spaCy pour la segmentation en phrases
        :param token_model: nom du modèle utilisé pour la tokenisation tiktoken
        :param max_tokens: taille maximale d'un chunk en tokens
        :param overlap: nombre de tokens à reculer pour le chevauchement
        """
        # On ne stocke que les NOMS des modèles, on ne charge rien !
        self.nlp_model = nlp_model
        self._nlp = None  # L'objet spaCy sera stocké ici, une fois chargé.

        try:
            self.enc = tiktoken.encoding_for_model(token_model)
        except Exception:
            self.enc = tiktoken.get_encoding(token_model)

        self.max_tokens = max_tokens
        self.overlap = overlap

    @property
    def nlp(self):
        """
        Propriété qui charge le modèle spaCy la toute première fois qu'il est appelé.
        Les appels suivants réutiliseront le modèle déjà chargé.
        """
        if self._nlp is None:
            print(f"--- LAZY LOADING: Chargement du modèle spaCy '{self.nlp_model}' ---")
            import spacy
            try:
                self._nlp = spacy.load(self.nlp_model)
            except OSError:
                print(f"Modèle SpaCy '{self.nlp_model}' non trouvé. Avez-vous exécuté :")
                print(f"python -m spacy download {self.nlp_model}")
                raise
        return self._nlp

    def split_into_chunks(
        self,
        text: str,
        file: str,
        page: int
    ) -> List[Chunk]:
        """
        Découpe un texte de page en chunks de tokens.
        """
        # Cette ligne va maintenant déclencher le chargement du modèle spaCy
        # via la propriété @property, mais seulement la première fois.
        sentences = [sent.text.strip() for sent in self.nlp(text).sents if sent.text.strip()]
        
        chunks: List[Chunk] = []
        current_tokens: List[int] = []

        for sent in sentences:
            token_ids = self.enc.encode(sent)
            if len(current_tokens) + len(token_ids) > self.max_tokens:
                chunk_text = self.enc.decode(current_tokens)
                chunks.append(Chunk(text=chunk_text, file=file, page=page))
                if self.overlap > 0:
                    current_tokens = current_tokens[-self.overlap:]
                else:
                    current_tokens = []
            current_tokens.extend(token_ids)

        if current_tokens:
            chunk_text = self.enc.decode(current_tokens)
            chunks.append(Chunk(text=chunk_text, file=file, page=page))

        return chunks

    def chunk_all(
        self,
        docs: List[Tuple[str, str, int]]
    ) -> List[Chunk]:
        """
        Applique split_into_chunks à chaque doc (texte, fichier, page).
        """
        all_chunks: List[Chunk] = []
        for text, file, page in docs:
            all_chunks.extend(self.split_into_chunks(text, file, page))
        return all_chunks