"""Prepare an instruct training checkpoint for evaluation.
https://huggingface.co/TrishanuDas/tayavision-instruct-checkpoint
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from models import save_for_inference
from pipeline.apply_lora import apply_lora
from src.processing import TinyAyaVisionProcessor

HF_REPO = "TrishanuDas/tayavision-instruct-checkpoint"
CHECKPOINT_FILE = "checkpoint_2465.pt"


def load_configs_from_hub(repo: str) -> tuple[TinyAyaVisionConfig, LoraAdapterConfig]:
    config_path = hf_hub_download(repo, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    model_cfg = hf_config["model_config"]
    if not model_cfg.get("cache_dir"):
        model_cfg["cache_dir"] = None

    vlm_config = TinyAyaVisionConfig(**model_cfg)
    lora_config = LoraAdapterConfig(**hf_config["lora_config"])

    return vlm_config, lora_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    logger.info(f"Loading configs from {HF_REPO}.")
    vlm_config, lora_config = load_configs_from_hub(HF_REPO)

    logger.info("Building model with LoRA adapters...")
    logger.info(f"  LLM:            {vlm_config.llm_model_name}")
    logger.info(f"  Vision encoder: {vlm_config.vision_model_name}")
    logger.info(f"  LoRA:           rank={lora_config.rank}, alpha={lora_config.lora_alpha}, "
                f"layers {lora_config.layers_to_transform[0]}-{lora_config.layers_to_transform[-1]}")

    model = apply_lora(vlm_config, lora_config)
    processor = TinyAyaVisionProcessor(config=vlm_config)
    model.setup_tokenizer(processor.tokenizer)
    model.to(device)

    logger.info(f"Downloading {CHECKPOINT_FILE} from {HF_REPO}.")
    checkpoint_path = hf_hub_download(HF_REPO, CHECKPOINT_FILE)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    logger.info("Loading projector weights.")
    model.multi_modal_projector.load_state_dict(checkpoint["projector"])

    logger.info("Loading LoRA adapter weights.")
    model.language_model.load_state_dict(checkpoint["lora_adapter"], strict=False)

    logger.info("Merging LoRA into base weights...")
    model.language_model = model.language_model.merge_and_unload()

    output_dir = Path(args.output_dir)
    logger.info(f"Saving model to {output_dir} ...")
    save_for_inference(model, processor, output_dir)
    logger.info("Successfully prepared checkpoint for evaluation")

if __name__ == "__main__":
    main()
