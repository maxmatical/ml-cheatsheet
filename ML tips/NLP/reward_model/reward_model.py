import torch
from torch import nn
from transformers import GPT2Model

class RewardModel(nn.Module):
    """
    Using GPT for reward model like in instructGPT
    """
    def __init__(self, base_model: GPT2Model):
        super().__init__()
        self.model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
    
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # bs, seq_len, hid_dim 
        logits = self.reward_head(out[:, -1, :]).squeeze(-1) # bs
        return logits
        