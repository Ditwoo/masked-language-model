import sys
import warnings
from catalyst.dl import registry
from .runner import Runner
from .experiment import Experiment

from .models import BertBasedMLM
from .callbacks import PerplexityCallback

if not sys.warnoptions:
    warnings.simplefilter("ignore")


registry.Model(BertBasedMLM)

registry.Callback(PerplexityCallback)
