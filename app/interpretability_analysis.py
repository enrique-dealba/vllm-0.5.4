import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionaries to hold activations and gradients
activation_dict = {}
neuron_grad_dict = {}


def record_activation_hook(layer_name):
    def hook(module, inputs, output):
        activation_dict[layer_name] = output.detach().cpu()

    return hook


def neuron_gradient_hook(layer_name):
    def hook(module, grad_input, grad_output):
        neuron_grad_dict[layer_name] = grad_output[0].detach().cpu()

    return hook


def get_model_and_tokenizer():
    model_name = os.getenv("LLM_MODEL_NAME", "tiiuae/Falcon3-7B-Instruct")

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    return model, tokenizer


def register_hooks(model):
    logger.info("Registering hooks...")
    hook_handles = []

    # Handle different model architectures
    if hasattr(model, "model"):
        # Falcon architecture
        layers = model.model.layers
        for idx, layer in enumerate(layers):
            # Register forward hook for activations
            handle = layer.mlp.register_forward_hook(
                record_activation_hook(f"mlp_{idx}")
            )
            hook_handles.append(handle)

            # Register backward hook for gradients
            handle = layer.mlp.register_full_backward_hook(
                neuron_gradient_hook(f"mlp_{idx}_grad")
            )
            hook_handles.append(handle)

            logger.info(f"Registered hooks for layer {idx}")
    elif hasattr(model, "transformer"):
        # LLaMA architecture
        for idx, layer in enumerate(model.transformer.h):
            handle = layer.mlp.register_forward_hook(
                record_activation_hook(f"mlp_{idx}")
            )
            hook_handles.append(handle)

            handle = layer.mlp.register_full_backward_hook(
                neuron_gradient_hook(f"mlp_{idx}_grad")
            )
            hook_handles.append(handle)

            logger.info(f"Registered hooks for layer {idx}")
    else:
        raise ValueError(
            f"Unsupported model architecture. Model structure: {model.__class__.__name__}"
        )

    return hook_handles


def analyze_model(input_text: str, save_dir: str = "/app/plots"):
    """Analyze model activations and gradients for a given input text.

    Args:
        input_text (str): Text to analyze
        save_dir (str): Directory to save visualization plots
    """
    logger.info("Starting model analysis...")

    # Clear previous data
    activation_dict.clear()
    neuron_grad_dict.clear()

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Store original mode
    was_training = model.training
    hook_handles = []

    try:
        # Register hooks before computation
        hook_handles = register_hooks(model)
        logger.info(f"Registered {len(hook_handles)} hooks")

        # Enable gradient computation temporarily
        model.train()

        # Tokenize and prepare input
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Forward and backward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        logger.info(f"Computed loss: {loss.item():.4f}")

        loss.backward()
        logger.info("Completed backward pass")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Plot activation distributions
        logger.info(f"Processing {len(activation_dict)} activation layers...")
        for layer_name, activations in activation_dict.items():
            plt.figure(figsize=(10, 6))
            act_values = activations.numpy().flatten()

            # Basic statistics
            mean = np.mean(act_values)
            std = np.std(act_values)

            plt.hist(act_values, bins=50, alpha=0.7)
            plt.axvline(mean, color="r", linestyle="dashed", linewidth=1)
            plt.axvline(mean + std, color="g", linestyle="dashed", linewidth=1)
            plt.axvline(mean - std, color="g", linestyle="dashed", linewidth=1)

            plt.title(
                f"Activation Distribution - {layer_name}\nμ={mean:.4f}, σ={std:.4f}"
            )
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = os.path.join(save_dir, f"{layer_name}_activations.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved activation plot: {filename}")

        # Plot gradient distributions
        logger.info(f"Processing {len(neuron_grad_dict)} gradient layers...")
        for layer_name, grads in neuron_grad_dict.items():
            plt.figure(figsize=(10, 6))
            grad_values = grads.numpy().flatten()

            # Basic statistics
            mean = np.mean(grad_values)
            std = np.std(grad_values)

            plt.hist(grad_values, bins=50, alpha=0.7)
            plt.axvline(mean, color="r", linestyle="dashed", linewidth=1)
            plt.axvline(mean + std, color="g", linestyle="dashed", linewidth=1)
            plt.axvline(mean - std, color="g", linestyle="dashed", linewidth=1)

            plt.title(
                f"Gradient Distribution - {layer_name}\nμ={mean:.4f}, σ={std:.4f}"
            )
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = os.path.join(save_dir, f"{layer_name}_gradients.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved gradient plot: {filename}")

        # Generate summary statistics
        summary_stats = {
            "activations": {
                layer: {
                    "mean": float(activation_dict[layer].mean()),
                    "std": float(activation_dict[layer].std()),
                    "min": float(activation_dict[layer].min()),
                    "max": float(activation_dict[layer].max()),
                }
                for layer in activation_dict
            },
            "gradients": {
                layer: {
                    "mean": float(neuron_grad_dict[layer].mean()),
                    "std": float(neuron_grad_dict[layer].std()),
                    "min": float(neuron_grad_dict[layer].min()),
                    "max": float(neuron_grad_dict[layer].max()),
                }
                for layer in neuron_grad_dict
            },
        }

        # Save summary statistics
        import json

        with open(os.path.join(save_dir, "analysis_summary.json"), "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info("Saved analysis summary")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup
        logger.info("Starting cleanup...")

        # Restore original mode
        model.train(was_training)

        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        logger.info("Removed all hooks")

        # Clear gradients
        model.zero_grad(set_to_none=True)
        logger.info("Cleared gradients")

        # Clear dictionaries
        activation_dict.clear()
        neuron_grad_dict.clear()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

        logger.info("Cleanup completed")

    return summary_stats


if __name__ == "__main__":
    sample_text = "Create a CatalogMaintenanceObjective for sensors RME04 and LMNT02 with U markings."
    try:
        analyze_model(sample_text)
        logger.info("Analysis completed successfully!")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
