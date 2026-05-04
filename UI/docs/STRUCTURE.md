# Chatbot Application - Project Structure

## Overview
This project has been refactored into a professional, modular structure for better maintainability, testability, and scalability.

## File Structure

```
UI/
├── main.py                 # Application entry point
├── config.py              # Configuration constants and settings
├── utils.py               # Utility functions (prompt building, response cleaning)
├── model_loader.py        # Model and tokenizer loading
├── conversation.py        # Conversation state management
├── gui.py                 # GUI implementation and event handlers
└── README.md              # Original project README
```

## Module Descriptions

### `main.py`
- **Purpose**: Entry point for the application
- **Key Functions**:
  - `main()`: Initializes the model and launches the GUI
- **Usage**: Run with `python main.py`

### `config.py`
- **Purpose**: Centralized configuration and constants
- **Contents**:
  - Model path and device configuration
  - System prompt
  - GUI styling (colors, fonts, geometry)
  - Generation parameters
- **Usage**: Import constants as needed: `from config import MODEL_PATH, DEVICE`

### `utils.py`
- **Purpose**: Utility functions for prompt handling and response processing
- **Key Functions**:
  - `build_prompt()`: Formats conversation history into model-compatible prompt
  - `clean_response()`: Normalizes and cleans model output
- **Usage**: `from utils import build_prompt, clean_response`

### `model_loader.py`
- **Purpose**: Handles model and tokenizer initialization
- **Key Functions**:
  - `load_model_and_tokenizer()`: Loads model from specified path with proper token configuration
- **Usage**: `from model_loader import load_model_and_tokenizer`

### `conversation.py`
- **Purpose**: Manages conversation state and history
- **Key Class**:
  - `ConversationManager`: Handles multiple conversations, switching, and summary generation
- **Key Methods**:
  - `new_conversation()`: Create new chat
  - `add_user_message()`: Add user input
  - `add_ai_message()`: Add AI response
  - `get_conversation_summary()`: Get display-friendly conversation summary
- **Usage**: `from conversation import ConversationManager`

### `gui.py`
- **Purpose**: GUI implementation and event handling
- **Key Class**:
  - `ChatbotGUI`: Main GUI class with all UI components and handlers
- **Key Methods**:
  - `_setup_layout()`: Initialize all GUI components
  - `_on_send_message()`: Handle message sending
  - `_generate_response()`: Generate AI response
  - `run()`: Start the application
- **Usage**: `from gui import ChatbotGUI`

## Benefits of This Structure

✅ **Modularity**: Each module has a single responsibility
✅ **Maintainability**: Easy to locate and modify specific functionality
✅ **Testability**: Functions and classes can be tested independently
✅ **Reusability**: Components can be imported and used elsewhere
✅ **Scalability**: Easy to add new features without affecting existing code
✅ **Configuration Management**: All settings in one place
✅ **Professional Convention**: Follows Python best practices

## Quick Start

1. Ensure all dependencies are installed:
   ```bash
   pip install torch transformers tkinter
   ```

2. Update the model path in `config.py` if necessary

3. Run the application:
   ```bash
   python main.py
   ```

## Configuration

To customize the application, edit `config.py`:
- **Model Path**: `MODEL_PATH`
- **GUI Appearance**: `GUI_*` constants
- **Generation Behavior**: `GENERATION_CONFIG`
- **System Prompt**: `SYSTEM_PROMPT`

## Extension Points

To extend functionality:

- **Add new utilities**: Add functions to `utils.py`
- **Change GUI styling**: Modify constants in `config.py` and `gui.py`
- **Add conversation features**: Extend `ConversationManager` class
- **Add new prompting strategies**: Create new functions in `utils.py`

---

*This refactored structure makes the codebase professional, maintainable, and ready for production use.*
