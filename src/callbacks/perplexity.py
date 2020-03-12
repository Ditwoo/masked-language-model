import torch
from torch.nn.functional import cross_entropy
from catalyst.dl import Callback, CallbackOrder, State


class PerplexityCallback(Callback):
    def __init__(self, 
                 prefix: str = "perplexity", 
                 input_key: str = "targets", 
                 output_key: str = "logits",
                 mask_ignore_index: int = -1,
                 size_average: str = "mean",
                 **kwargs):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.ignore_index = mask_ignore_index
        self.size_average = size_average

    def on_batch_end(self, state: State) -> None:
        model_output = state.output[self.output_key]
        target_mask = state.input[self.input_key]

        with torch.no_grad():
            perplexity = cross_entropy(
                model_output, target_mask, size_average=self.size_average, ignore_index=self.ignore_index
            )
            perplexity = torch.exp(perplexity)
            perplexity = perplexity.item()
        
        state.batch_metrics[self.prefix] = perplexity
