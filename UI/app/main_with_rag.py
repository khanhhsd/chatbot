"""
Main entry point for the Chatbot application with RAG support.

This module initializes the model and starts the GUI.

RAG Integration:
- To use RAG (Retrieval-Augmented Generation):
  1. Set USE_RAG = True in config.py
  2. Install dependencies: pip install sentence-transformers scikit-learn
  3. Uncomment the gui import below
  4. Run this script
"""
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path whether run as script or module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_loader import load_model_and_tokenizer
from core.config import USE_RAG


def main():
    """Initialize model and launch the GUI application."""
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    # Choose GUI based on RAG setting
    if USE_RAG:
        try:
            # RAG-enabled GUI with context display
            from app.gui_rag import ChatbotGUI
            print("Launching with RAG enabled...")
        except ImportError as e:
            print(f"RAG import failed: {e}")
            print("Falling back to standard GUI...")
            from app.gui import ChatbotGUI
    else:
        # Standard GUI without RAG
        from app.gui import ChatbotGUI
        print("Launching standard chatbot (RAG disabled)...")
    
    # Initialize and run GUI
    gui = ChatbotGUI(tokenizer, model)
    gui.run()


if __name__ == "__main__":
    main()
