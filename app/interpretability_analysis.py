import logging
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dictionaries to hold activations and gradients.
activation_dict = {}
neuron_grad_dict = {}


def record_activation_hook(layer_name):
    """Hook to record forward-pass activations."""

    def hook(module, inputs, output):
        activation_dict[layer_name] = output.detach().cpu()

    return hook


def neuron_gradient_hook(layer_name):
    """Backward hook to record per-neuron gradients."""

    def hook(module, grad_input, grad_output):
        neuron_grad_dict[layer_name] = grad_output[0].detach().cpu()

    return hook


def register_hooks(transformer_model):
    """Register hooks on each Transformer block's MLP.
    This will capture both activations and neuron-level gradients.
    """
    for idx, block in enumerate(transformer_model.transformer.h):
        # Hook the MLP output to capture activations.
        block.mlp.register_forward_hook(record_activation_hook(f"mlp_{idx}"))
        # Register a backward hook on the MLP to capture gradients.
        block.mlp.register_backward_hook(neuron_gradient_hook(f"mlp_{idx}_grad"))


def get_underlying_model():
    model_name = os.getenv("LLM_MODEL_NAME", "tiiuae/Falcon3-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


# Update tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(
    os.getenv("LLM_MODEL_NAME", "tiiuae/Falcon3-7B-Instruct"), use_fast=False
)


def run_forward(input_text: str):
    """Run a forward pass on the underlying model to capture activations."""
    transformer_model = get_underlying_model()
    register_hooks(transformer_model)
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    print("Captured Activations:")
    for layer_name, act in activation_dict.items():
        print(f"  {layer_name}: {act.shape}")
    return outputs


def compute_gradients(input_text: str):
    """Perform a forward and backward pass to compute gradients.
    This toy example uses the input tokens as labels (next-token prediction).
    """
    transformer_model = get_underlying_model()
    transformer_model.train()  # Enable gradient computation.
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = transformer_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    transformer_model.zero_grad()
    loss.backward()

    grad_magnitudes = {}
    for name, param in transformer_model.named_parameters():
        if param.grad is not None:
            grad_magnitudes[name] = param.grad.data.norm().item()
    return grad_magnitudes


def plot_neuron_gradient_distribution(save_plots=False, save_dir="/app/plots"):
    logger.info(
        f"Starting plot generation. Save plots: {save_plots}, Save dir: {save_dir}"
    )
    logger.info(f"Number of gradient layers: {len(neuron_grad_dict)}")

    if not neuron_grad_dict:
        logger.error(
            "No gradients were captured! Check if hooks were properly registered"
        )
        return

    for layer_name, grad_tensor in neuron_grad_dict.items():
        logger.info(f"Processing layer: {layer_name}")
        try:
            plt.figure(figsize=(6, 3))
            grad_values = grad_tensor.numpy().flatten()
            plt.hist(grad_values, bins=50, alpha=0.7)
            plt.title(f"Gradient Distribution for {layer_name}")
            plt.xlabel("Gradient value")
            plt.ylabel("Frequency")
            plt.tight_layout()

            if save_plots:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(
                    save_dir, f"{layer_name}_gradient_distribution.png"
                )
                plt.savefig(filename)
                logger.info(f"Saved plot to {filename}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.error(
                f"Error processing layer {layer_name}: {str(e)}", exc_info=True
            )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting analysis...")
        sample_text = "Create a CatalogMaintenanceObjective for sensors RME04 and LMNT02 with U markings."
        logger.info(f"Computing gradients for text: {sample_text}")

        grad_magnitudes = compute_gradients(sample_text)
        logger.info(f"Computed gradient magnitudes: {grad_magnitudes}")

        logger.info("Generating plots...")
        plot_neuron_gradient_distribution(save_plots=True, save_dir="/app/plots")
        logger.info("Analysis complete!")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
