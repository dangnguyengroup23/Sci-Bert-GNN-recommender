import torch
import torch.nn as nn
from transformers import AutoModel

class BertProjector(nn.Module):
    def __init__(self, model_name, num_classes, proj_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.bert.config, 'hidden_size', 768)
        self.classifier = nn.Linear(hidden, num_classes)
        self.projection = nn.Linear(hidden, proj_dim)

    def forward(self, input_ids, attention_mask, project=True):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(last).float()
        summed = (last * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1e-9)
        pooled = summed / lengths
        logits = self.classifier(pooled)
        projected = self.projection(pooled) if project else pooled
        return logits, projected