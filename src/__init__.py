from catalyst.dl import registry
from .runner import Runner

from .models import BertBasedMLM
from .callbacks import PerplexityCallback


registry.Model(BertBasedMLM)

registry.Callback(PerplexityCallback)
