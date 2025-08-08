# src/streamlit_chat.py

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement (.env) au démarrage
load_dotenv()

# Imports des composants nécessaires
from rag_agent.config import load_settings, Settings
from rag_agent.pipeline import RAGPipeline
from rag_agent.components.chunking import Chunker

# Configuration de la page Streamlit
st.set_page_config(page_title="Assistant d'Astreinte", layout="wide", initial_sidebar_state="auto")
st.title("💬 Assistant Technique d'Astreinte")
st.caption("Posez vos questions sur la documentation technique. Je chercherai la réponse pour vous.")

# --- Mise en cache des objets "LÉGERS" et "SÛRS" ---

@st.cache_resource
def get_main_config() -> Settings:
    """Charge la configuration principale une seule fois."""
    return load_settings(Path("settings.json"))

@st.cache_resource
def get_pipeline(_settings: Settings) -> RAGPipeline:
    """
    Crée et met en cache le RAGPipeline (sans le Chunker).
    C'est un objet que Streamlit peut mettre en cache sans problème.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral non configurée.")
        return None
    return RAGPipeline(settings=_settings, api_key=api_key)


# --- ORCHESTRATION PRINCIPALE ---
try:
    # 1. Obtenir les composants mis en cache (les objets sûrs)
    main_settings = get_main_config()
    pipeline = get_pipeline(main_settings)

    # 2. Gérer l'objet "LOURD" (le Chunker) avec st.session_state
    # C'est la solution définitive qui contourne le bug de cache de Streamlit.
    if 'chunker' not in st.session_state:
        with st.spinner("Initialisation du composant d'analyse de texte (une seule fois)..."):
            st.session_state.chunker = Chunker(
                nlp_model=main_settings.nlp_model,
                token_model=main_settings.tokenizer_encoding,
                max_tokens=main_settings.chunk_max_tokens,
                overlap=main_settings.chunk_overlap
            )

    # 3. Préchauffer l'index (si nécessaire)
    if "index_initialized" not in st.session_state:
        with st.spinner("Préparation de la base de connaissances..."):
            # On passe l'objet chunker depuis la session_state
            pipeline.build_or_update(chunker=st.session_state.chunker, mode="auto")
        st.session_state.index_initialized = True
        st.success("Assistant prêt !")
        st.rerun()

    # 4. Logique de l'interface de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Je cherche dans la documentation..."):
                # On passe l'objet chunker depuis la session_state à chaque appel
                response = pipeline.answer(user_input, chunker=st.session_state.chunker, update_mode="auto")
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Une erreur critique est survenue : {e}")
    st.exception(e)


#streamlit run src/streamlit_chat.py

#Comment accéder physiquement aux serveurs PDA et WINCC ?

#comment accéder au poste chorus équipé de Reflexion X? 