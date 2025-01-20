
import json

# Save model weights to a file
def save_model_weights(weights, filepath):
    with open(filepath, 'w') as f:
        json.dump(weights, f)
    print(f"Model weights saved to {filepath}.")

# Load model weights from a file
def load_model_weights(filepath):
    with open(filepath, 'r') as f:
        weights = json.load(f)
    print(f"Model weights loaded from {filepath}.")
    return weights
        