"""
Initializes the model and starts the GUI.
"""
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path whether run as script or module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_loader import load_model_and_tokenizer
from app.gui import ChatbotGUI


def main():
    """Initialize model and launch the GUI application."""
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    # Initialize and run GUI
    gui = ChatbotGUI(tokenizer, model)
    gui.run()


if __name__ == "__main__":
    main()
