"""
Model and tokenizer loading utilities.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import MODEL_PATH, DEVICE


def load_model_and_tokenizer():
    """
    Load the model and tokenizer from the specified path.
    
    Returns:
        tuple: (tokenizer, model) loaded and configured for inference
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Ensure eos token is set
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": ""})
        model.resize_token_embeddings(len(tokenizer))
    
    # Ensure pad token is set and resize embeddings
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model
