from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from transformers import BertTokenizer
from .datasets import LinesDataset, MLMCollateFn


class Experiment(ConfigExperiment):

    def get_datasets(self, 
                     stage: str, 
                     train: str, 
                     valid: str, 
                     tokenizer: str, 
                     mask_ignore_index: int = -1, 
                     **kwargs) -> OrderedDict:
        tokenizer = BertTokenizer.from_pretrained(tokenizer)

        datasets = OrderedDict()
        datasets["train"] = dict(
            dataset=LinesDataset(
                file=train, 
                tokenizer=tokenizer,
                masking_pcnt=0.15,
                mask_with_random_word_pcnt=0.1
            ),
            collate_fn=MLMCollateFn(
                pad_token_id=tokenizer.pad_token_id,
                mask_ignore_index=mask_ignore_index,
                post_pad=True,
            ),
            shuffle=True,
        )
        print(f" * Num records in train dataset - {len(datasets['train']['dataset'])}")

        datasets["valid"] = dict(
            dataset=LinesDataset(
                file=train, 
                tokenizer=tokenizer,
                masking_pcnt=0.15,
                mask_with_random_word_pcnt=0.1
            ),
            collate_fn=MLMCollateFn(
                pad_token_id=tokenizer.pad_token_id,
                mask_ignore_index=mask_ignore_index,
                post_pad=True,
            ),
            shuffle=False,
        )
        print(f" * Num records in valid dataset - {len(datasets['valid']['dataset'])}")

        return datasets
