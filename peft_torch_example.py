import torch 
from peft import inject_adapter_in_model, LoraConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

lora_config = LoraConfig(
    # lora_alpha=16,
    # lora_dropout=0.1,
    r=64,
    # bias="none",
    target_modules=["linear"],
)

model = DummyModel()

print("\nBefore inject: \n", model)

model = inject_adapter_in_model(lora_config, model)

print("\nAfter inject: \n", model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
