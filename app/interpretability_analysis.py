import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionaries to hold activations and gradients
activation_dict = {}
neuron_grad_dict = {}


def llama_record_activation_hook(layer_name):
    def hook(module, inputs, output):
        # Detach and move to CPU to avoid GPU memory issues.
        activation_dict[layer_name] = output.detach().cpu()

    return hook


def llama_neuron_gradient_hook(layer_name):
    def hook(module, grad_input, grad_output):
        # grad_output is a tuple; take the first element.
        neuron_grad_dict[layer_name] = grad_output[0].detach().cpu()

    return hook


def register_llama_hooks(model):
    hook_handles = []
    # Assume the underlying model has an attribute 'model' which is the LlamaModel.
    llama_model = model.model
    if not hasattr(llama_model, "layers"):
        raise ValueError("No layers attribute found in the underlying LlamaModel.")

    for idx, layer in enumerate(llama_model.layers):
        # Register hooks only if the layer has an MLP submodule.
        if hasattr(layer, "mlp"):
            # Use our recording hooks instead of lambda logging functions.
            hook_fwd = layer.mlp.register_forward_hook(
                llama_record_activation_hook(f"mlp_{idx}")
            )
            hook_handles.append(hook_fwd)
            hook_bwd = layer.mlp.register_full_backward_hook(
                llama_neuron_gradient_hook(f"mlp_{idx}")
            )
            hook_handles.append(hook_bwd)
            logger.info(f"Registered hooks for layer {idx} MLP")
    return hook_handles


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
    """Analyze model activations for a given input text.

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

    # Put model in eval mode
    model.eval()
    hook_handles = []

    try:
        # Register hooks before computation
        hook_handles = register_hooks(model)
        logger.info(f"Registered {len(hook_handles)} hooks")

        # Tokenize and prepare input with attention mask
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Debug: Log input_ids dtype and shape before casting
        logger.info(
            f"Input IDs dtype before casting: {inputs['input_ids'].dtype}, shape: {inputs['input_ids'].shape}"
        )

        # Ensure input_ids are of type long (required for embedding)
        inputs["input_ids"] = inputs["input_ids"].long()
        logger.info(f"Input IDs dtype after casting: {inputs['input_ids'].dtype}")
        logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Forward pass only
        with torch.no_grad():
            temperature = float(os.getenv("TEMPERATURE", 0.2))
            max_tokens = int(os.getenv("MAX_TOKENS", 8192))

            generated_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            generated_text = tokenizer.decode(
                generated_ids[0][input_length:], skip_special_tokens=True
            )
            final_output = generated_text

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Prepare comprehensive data dictionary
        summary_stats = {
            "metadata": {
                "input_text": input_text,
                "full_prompt": input_text,
                "final_output": str(final_output),
                "vllm_langchain_pipeline": False,
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
            # "gradients": {
            #     layer: {
            #         "summary": {
            #             "mean": float(neuron_grad_dict[layer].float().mean()),
            #             "std": float(neuron_grad_dict[layer].float().std()),
            #             "min": float(neuron_grad_dict[layer].float().min()),
            #             "max": float(neuron_grad_dict[layer].float().max()),
            #         },
            #         "histogram": compute_histogram(
            #             neuron_grad_dict[layer].float().numpy().flatten(), bins=20
            #         ),
            #     }
            #     for layer in neuron_grad_dict
            # },
        }

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup
        logger.info("Starting cleanup...")
        for handle in hook_handles:
            handle.remove()
        logger.info("Removed all hooks")
        activation_dict.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
        logger.info("Cleanup completed")

    return summary_stats
