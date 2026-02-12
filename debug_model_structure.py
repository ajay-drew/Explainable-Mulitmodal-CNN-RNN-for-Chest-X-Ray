import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from config import MODEL_NAME

def inspect_model():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    print("\n--- Model Configuration ---")
    print(model.config)
    
    print("\n--- ViT Structure ---")
    print(model.vit)
    
    # Check expected input size
    print(f"\nExpected Input Size: {model.config.image_size}")
    print(f"Patch Size: {model.config.patch_size}")
    
    # Calculate expected tokens
    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    print(f"Calculated Patches: {num_patches}")
    print(f"Expected Sequence Length (with CLS): {num_patches + 1}")

    # Run a dummy input to verify
    dummy_input = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    outputs = model.vit(dummy_input)
    print(f"\nLast Hidden State Shape: {outputs.last_hidden_state.shape}")

if __name__ == "__main__":
    inspect_model()
