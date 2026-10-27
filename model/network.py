"""DeBERTa encoder with a five-class rating classifier (classes 1..5)."""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
class RatingModel(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder=encoder; h=encoder.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(h,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,5))
        nn.init.normal_(self.head[-1].weight,std=.01); nn.init.zeros_(self.head[-1].bias)
    @classmethod
    def pretrained(cls,name,revision=None,initial_rating=None): return cls(AutoModel.from_pretrained(name,revision=revision))
    @classmethod
    def from_checkpoint_config(cls,path): return cls(AutoModel.from_config(AutoConfig.from_pretrained(path)))
    def forward(self,input_ids,attention_mask):
        seq=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1).float(); pooled=(seq.float()*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(pooled)
