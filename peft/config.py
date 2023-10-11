from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class LoraConfig:
    r: int = field(default=8, metadata={"help": "Lora attention dimension."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )