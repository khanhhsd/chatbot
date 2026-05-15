
This is a local AI project with a focus on a Tkinter-based user interface, a model data folder, and a dataset archive for nutrition or food-related tasks.

## Repository structure

- `DataBase/`
  - `archive/` contains dataset CSV files and metadata for food-related data.
- `Model/`
  - Contains model files, tokenizer assets, and model configuration for a pretrained language model.
- `UI/`
  - `app/` contains Python GUI source code for the TTCS application.
  - `assets/` can be used for UI resources or supporting files.
  - `docs/` contains documentation and guides for the RAG/UI portion of the project.

## Getting started

1. Open the workspace in VS Code.
2. Install any required Python dependencies for the UI app if needed.
   pip install sentence-transformers
  pip install scikit-learn
3. Run the GUI application `UI/app/main_with_rag.py`.

## Notes

- The `UI` folder includes docs for working with RAG, UI components, and tests.
- The `Model` folder appears to hold a Hugging Face-style model with tokenizers and config.
- `DataBase/archive/FINAL FOOD DATASET/` contains CSV datasets and metadata for analysis.

## Recommended entry points
- `UI/app/main_with_rag.py`: start the app with RAG support if configured.

