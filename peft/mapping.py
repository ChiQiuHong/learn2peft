import torch

from .config import LoraConfig

from .tuners import LoraModel

def inject_adapter_in_model(
    lora_config: LoraConfig, model: torch.nn.Module
) -> torch.nn.Module:
    peft_model = LoraModel(model, lora_config)

    return peft_model.model