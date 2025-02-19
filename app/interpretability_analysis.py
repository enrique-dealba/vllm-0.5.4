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
        logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Forward pass only
        with torch.no_grad():
            # outputs = model(**inputs)

            # Generate text output
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
            final_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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


if __name__ == "__main__":
    # sample_text = "Create a CatalogMaintenanceObjective for sensors RME04 and LMNT02 with U markings."

    input_text_1 = (
        "You are a helpful AI assistant that always responds in valid JSON format."
        "\\n\\nFormat your response according to this schema:\\n"
        "The output should be formatted as a JSON instance that conforms to the JSON "
        "schema below.\n\n"
        'As an example, for the schema {"properties": {"foo": {"title": "Foo", '
        '"description": "a list of strings", "type": "array", "items": '
        '{"type": "string"}}}, "required": ["foo"]}\n'
        'the object {"foo": ["bar", "baz"]} is a well-formatted instance of the '
        'schema. The object {"properties": {"foo": ["bar", "baz"]}} is not '
        "well-formatted.\n\n"
        "Here is the output schema:\n```\n"
        '{"properties": {"objective_name": {"description": "Type of objective '
        "being specified. Choose from: CatalogMaintenanceObjective, "
        "PeriodicRevisitObjective, SearchObjective, DataEnrichmentObjective, "
        "GeodssRevisitObjective, SensorCheckoutObjective, SingleIntentObjective, "
        'UctObservationObjective, BaselineAutonomyObjective", "title": '
        '"Objective Name", "type": "string"}}, "required": '
        '["objective_name"]}\n```'
        "\\n\\nRemember:\\n"
        "1. Your response MUST be valid JSON\\n"
        "2. Do not include any explanatory text outside the JSON\\n"
        "3. Ensure all required fields are included\\n"
        "4. Use the exact field names specified\\n\\n"
        "User Query: Create a CatalogMaintenanceObjective for sensors RME04 and "
        "LMNT02 with U markings, TEST mode, priority 12, patience of 10 mins, end "
        "time offset of 25 mins, visibility check false. Start at 2024-05-21 "
        "19:20:00+00:00, end at 2024-05-21 22:30:00+00:00. Use "
        "RATE_TRACK_SIDEREAL tracking in LEO regime. RSO ID list includes "
        "12445,67889\\n\\n"
        "JSON Response:"
    )

    input_text_2 = (
        "You are a helpful AI assistant that always responds in valid JSON format."
        "\\n\\nFormat your response according to this schema:\\n"
        "The output should be formatted as a JSON instance that conforms to the JSON "
        "schema below.\n\n"
        'As an example, for the schema {"properties": {"foo": {"title": "Foo", '
        '"description": "a list of strings", "type": "array", "items": '
        '{"type": "string"}}}, "required": ["foo"]}\n'
        'the object {"foo": ["bar", "baz"]} is a well-formatted instance of the '
        'schema. The object {"properties": {"foo": ["bar", "baz"]}} is not '
        "well-formatted.\n\n"
        "Here is the output schema:\n```\n"
        '{"properties": {"classification_marking": {"description": "Classification '
        'level of objective intents. Choose from: U, C, S, TS, U//FOUO", '
        '"title": "Classification Marking", "type": "string"}, '
        '"data_mode": {"description": "String type for the Machina Common DataModeType. '
        'Choose from: TEST, REAL, SIMULATED, EXERCISE", "title": "Data Mode", '
        '"type": "string"}, "collect_request_type": {"default": "RATE_TRACK_SIDEREAL", '
        '"description": "Collect request type of tracking type. Choose from: RATE_TRACK, '
        "SIDEREAL, RATE_TRACK_SIDEREAL. Defaults to RATE_TRACK_SIDEREAL. Note: do NOT "
        'confuse RATE_TRACK with RATE_TRACK_SIDEREAL", "title": "Collect Request Type", '
        '"type": "string"}, "orbital_regime": {"anyOf": [{"type": "string"}, '
        '{"type": "null"}], "default": null, "description": "Orbital regime '
        "classification for this catalog maintenance objective. Choose from: LEO, MEO, "
        'GEO, XGEO", "title": "Orbital Regime"}, "patience_minutes": {"default": 30, '
        '"description": "Amount of time in minutes to wait before assuming an intent '
        'has failed, defaults to 30", "title": "Patience Minutes", "type": "integer"}, '
        '"end_time_offset_minutes": {"default": 20, "description": "Number of minutes '
        'into the future to schedule this intent, defaults to 20", '
        '"title": "End Time Offset Minutes", "type": "integer"}, '
        '"priority": {"default": 999, "description": "Priority level for scheduling '
        '(higher numbers indicate lower priority, defaults to 999)", "title": "Priority", '
        '"type": "integer"}, "sensor_name_list": {"anyOf": [{"items": {"type": "string"}, '
        '"type": "array"}, {"type": "null"}], "default": null, '
        '"description": "List of sensor names to be used", "title": "Sensor Name List"}, '
        '"rso_id_list": {"anyOf": [{"items": {"type": "string"}, "type": "array"}, '
        '{"type": "null"}], "default": [], "description": "Optional list of RSO IDs", '
        '"title": "Rso Id List"}, "objective_start_time": {"anyOf": [{"format": '
        '"date-time", "type": "string"}, {"type": "null"}], "default": null, '
        '"description": "Start time of the objective in ISO 8601 format with timezone", '
        '"title": "Objective Start Time"}, "objective_end_time": {"anyOf": [{"format": '
        '"date-time", "type": "string"}, {"type": "null"}], "default": null, '
        '"description": "End time of the objective in ISO 8601 format with timezone", '
        '"title": "Objective End Time"}, "objective_uuid": {"anyOf": [{"type": "string"}, '
        '{"type": "null"}], "default": null, "description": "Objective UUID generated '
        'by the belief state", "title": "Objective Uuid"}, "frame_type": {"default": '
        '"LIGHT", "description": "Frame type for the objective. Default is LIGHT", '
        '"title": "Frame Type", "type": "string"}, "binning": {"anyOf": [{"type": '
        '"integer"}, {"type": "null"}], "default": null, "description": '
        '"ImagerInstrument intent parameters", "title": "Binning"}, '
        '"visibility_check": {"default": false, "description": "Flag to determine RSO '
        'visibility before intent generation", "title": "Visibility Check", '
        '"type": "boolean"}, "objective_name": {"anyOf": [{"type": "string"}, '
        '{"type": "null"}], "default": "CatalogMaintenanceObjective", '
        '"description": "Name for this objective. Defaults to '
        '\'CatalogMaintenanceObjective\'", "title": "Objective Name"}}, '
        '"required": ["classification_marking", "data_mode"]}\n```'
        "\\n\\nRemember:\\n"
        "1. Your response MUST be valid JSON\\n"
        "2. Do not include any explanatory text outside the JSON\\n"
        "3. Ensure all required fields are included\\n"
        "4. Use the exact field names specified\\n\\n"
        "User Query: Create a CatalogMaintenanceObjective for sensors RME04 and "
        "LMNT02 with U markings, TEST mode, priority 12, patience of 10 mins, end "
        "time offset of 25 mins, visibility check false. Start at 2024-05-21 "
        "19:20:00+00:00, end at 2024-05-21 22:30:00+00:00. Use "
        "RATE_TRACK_SIDEREAL tracking in LEO regime. RSO ID list includes "
        "12445,67889\\n\\n"
        "JSON Response:"
    )

    stats_1 = {}
    stats_2 = {}

    # Part 1
    try:
        stats_1 = analyze_model(input_text_1)
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)

    # Part 2
    try:
        stats_2 = analyze_model(input_text_2)
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)

    save_dir = "/app/plots"
    current_time = datetime.now()
    timestamp = current_time.strftime("%m%d%Y_%H%M")
    filename = f"vanilla_{timestamp}.json"
    combined_data = {"part_1": stats_1, "part_2": stats_2}

    # Save comprehensive data
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(combined_data, f, indent=2)
    logger.info(f"Saved combined analysis data to {filename}")
    logger.info("SUCCESS: Vanilla analysis completed successfully!")
