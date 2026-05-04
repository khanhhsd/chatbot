"""
Configuration and constants for the chatbot application.
"""
import torch

# Model configuration
MODEL_PATH = r"D:\hocj\AI\TTCS\Model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# System prompt for the AI
SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in food and nutrition. Answer questions directly and concisely.\n"
    "Conversation format MUST be exactly:\n"
    "You: [user input]\n"
    "AI: [assistant response]\n\n"
    "Rules:\n"
    "- Never invent new questions or user turns.\n"
    "- All the information you need is in the retrieved context. Do not make assumptions beyond it.\n"
    "- CRITICAL: Do not invent values. If there is [CALCULATED NUTRITION DATA], use ONLY those exact values in your response.\n"
    "- NEVER show placeholders like <insert number>, <PROTEIN VALUE>, ##(calorie), or similar.\n"
    "- ALWAYS show complete numeric values with units (e.g., '42.5g', '150 kcal').\n"
    "Below is the user's question and the calculated nutrition data (if any). Use the CALCULATED NUTRITION DATA to answer the user question.\n"
    "Do not make any calculations yourself - only use the values provided in [CALCULATED NUTRITION DATA].\n"
    "If calculated nutrition data is provided, base your entire response on those values.\n"
)

# GUI configuration
GUI_WINDOW_TITLE = "UI"
GUI_WINDOW_GEOMETRY = "1600x900"
GUI_BG_COLOR = "black"
GUI_FG_COLOR = "white"
GUI_FONT_NAME = "Comic Sans MS"
GUI_FONT_SIZE = 13

# Text input configuration
INPUT_FRAME_HEIGHT = 3

# History display
HISTORY_MAX_DISPLAY_LENGTH = 30

# Model generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 400,
    "do_sample": False,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 50,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.5,
}

# RAG (Retrieval-Augmented Generation) Configuration
USE_RAG = True  # Enable/disable RAG
RAG_DATABASE_PATH = r"D:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET"
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for embeddings
RAG_TOP_K = 50  # Number of documents to retrieve
RAG_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# Vector Database Option (for production use)
# Options: "faiss" (fast), "milvus" (scalable), "simple" (current in-memory approach)
RAG_VECTOR_DB_TYPE = "simple"
