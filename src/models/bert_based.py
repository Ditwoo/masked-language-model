import torch
import torch.nn as nn
from transformers import BertForMaskedLM


class BertBasedMLM(nn.Module):
    def __init__(self, pretrain: str = 'bert-base-uncased'):
        self.core = BertForMaskedLM.from_pretrained(pretrain)

    def forward(self, input_ids, token_type_ids=None):
        out = self.core(input_ids=input_ids, token_type_ids=token_type_ids)[0]
        num_classes = out.size(2)
        return out.view(-1, num_classes)
