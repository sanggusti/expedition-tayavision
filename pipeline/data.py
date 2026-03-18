import json
import torch

from pathlib import Path
from PIL import Image

from config.model_config import TinyAyaVisionConfig
from src.processing import TinyAyaVisionProcessor

class AlignmentDataset(torch.utils.data.Dataset):
    """
    Dataset for aligning vision encoder w/LLM backbone via a learned connector.
    LLaVA-Pretrain dataset with image-caption pairs.
    """
    def __init__(
        self,
        config: TinyAyaVisionConfig,
        dataset_name: str = "liuhaotian/LLaVA-Pretrain",
        data_dir: str = "data/llava-pretrain",
    ):
        self.data_dir = Path(data_dir)
        print(f"Loading dataset from {self.data_dir / 'blip_laion_cc_sbu_558k.json'}...")
        with open(self.data_dir / "blip_laion_cc_sbu_558k.json", "r") as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} examples")
        self.processor = TinyAyaVisionProcessor(
            config=config,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = self.data_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        prompt = item["conversations"][0]["value"]
        response = item["conversations"][1]["value"]

        tokenizer = self.processor.tokenizer
        if tokenizer.chat_template is not None:
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full_text = prompt + response
            prompt_text = prompt

        processed = self.processor(
            images=image,
            text=full_text,
        )
        input_ids = processed["input_ids"].squeeze(0)
        attention_mask = processed["attention_mask"].squeeze(0)
        pixel_values = processed["pixel_values"].squeeze(0)

        processed_prompt = self.processor(
            text=prompt_text,
            image_grid_hws=processed.get("image_grid_hws"),
        )
        num_prompt_tokens = processed_prompt["input_ids"].shape[-1]
        labels = input_ids.clone()
        labels[:num_prompt_tokens] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        if "image_grid_hws" in processed:
            result["image_grid_hws"] = processed["image_grid_hws"].squeeze(0)
        return result

def collate_fn(
    batch,
    pad_token_id: int,
):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    # MoonViT: variable tile counts per image → concatenate along dim 0.
    # SigLIP: fixed-size tensors → stack into a batch.
    if "image_grid_hws" in batch[0]:
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    else:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }
    if "image_grid_hws" in batch[0]:
        result["image_grid_hws"] = torch.stack([item["image_grid_hws"] for item in batch])
    return result

    
