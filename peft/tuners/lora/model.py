import re

import torch
from torch import nn

from .layer import Linear


class LoraModel(nn.Module):
    def __init__(self, model, lora_config):
        super().__init__()
        self.model = model
        self.lora_config = lora_config

        self.inject_adapter(self.model)


    def inject_adapter(self, model: nn.Module):
        lora_config = self.lora_config
        key_list = [key for key, _ in model.named_modules()]

        for key in key_list:
            if not check_target_module_exists(lora_config, key):
                continue

            parent, target, target_name = _get_submodules(model, key)
            self._create_and_replace(lora_config, target, target_name, parent, key)

    def _create_and_replace(
            self,
            lora_config,
            target,
            target_name,
            parent,
            current_key,
        ):
        new_module = self._create_new_module(lora_config, target)
        
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

    def _create_new_module(self, lora_config, target):
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
        # new_module = Linear(in_features, out_features)
        new_module = nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 10),
        )

        return new_module


def check_target_module_exists(config, key: str) -> bool:
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules
    
    return target_module_found


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)

    return parent, target, target_name