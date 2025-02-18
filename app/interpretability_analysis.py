import json
import logging
import os
from datetime import datetime

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


def compute_histogram(values, bins=20):
    hist, bin_edges = np.histogram(values, bins=bins)
    return {"bin_edges": bin_edges.tolist(), "counts": hist.tolist()}


def analyze_model(input_text: str, save_dir: str = "/app/plots"):
    """Analyze model activations and gradients for a given input text.

    Args:
        input_text (str): Text to analyze
        save_dir (str): Directory to save analysis data
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

        # Prepare comprehensive data dictionary
        summary_stats = {
            "metadata": {
                "input_text": input_text,
                "sequence_length": int(inputs["input_ids"].shape[1]),
                "loss": float(loss.item()),
            },
            "activations": {
                layer: {
                    "summary": {
                        "mean": float(activation_dict[layer].float().mean()),
                        "std": float(activation_dict[layer].float().std()),
                        "min": float(activation_dict[layer].float().min()),
                        "max": float(activation_dict[layer].float().max()),
                    },
                    "histogram": compute_histogram(
                        activation_dict[layer].float().numpy().flatten(), bins=20
                    ),
                }
                for layer in activation_dict
            },
            "gradients": {
                layer: {
                    "summary": {
                        "mean": float(neuron_grad_dict[layer].float().mean()),
                        "std": float(neuron_grad_dict[layer].float().std()),
                        "min": float(neuron_grad_dict[layer].float().min()),
                        "max": float(neuron_grad_dict[layer].float().max()),
                    },
                    "histogram": compute_histogram(
                        neuron_grad_dict[layer].float().numpy().flatten(), bins=20
                    ),
                }
                for layer in neuron_grad_dict
            },
        }

        # Generate timestamp for filename
        current_time = datetime.now()
        timestamp = current_time.strftime("%m%d%Y_%H%M")
        filename = f"interp_{timestamp}.json"

        # Save comprehensive data
        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Saved analysis data to {filename}")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup
        logger.info("Starting cleanup...")
        model.train(was_training)
        for handle in hook_handles:
            handle.remove()
        logger.info("Removed all hooks")
        model.zero_grad(set_to_none=True)
        logger.info("Cleared gradients")
        activation_dict.clear()
        neuron_grad_dict.clear()
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
