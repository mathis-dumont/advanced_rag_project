# src/rag_agent/config.py
from pathlib import Path
import json
from pydantic import BaseModel, field_validator, model_validator

class Settings(BaseModel):
    data_dir: Path
    pdf_output_dir: Path
    db_dir: Path
    nlp_model: str

    tokenizer_encoding: str

    embedding_model: str
    chat_model: str
    chunk_max_tokens: int
    chunk_overlap: int
    k: int
    window: int
    process_images: bool = True

    @model_validator(mode='after')
    def create_directories(self) -> 'Settings':
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        return self
    

def load_settings(path: Path = Path("settings.json")) -> Settings:
    """Charge les paramètres depuis un fichier JSON et les valide avec Pydantic."""
    if not path.exists():
        raise FileNotFoundError(f"Le fichier de configuration '{path}' est introuvable.")
    
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return Settings.model_validate(data)