import numpy as np
from torch.utils.data import Dataset
from transformers import BasicTokenizer
from typing import List, Tuple


class TokensDataset(Dataset):
    def __init__(self, 
                 texts: List[str], 
                 tokenizer: BasicTokenizer,
                 masking_pcnt: float = 0.15, 
                 mask_with_random_word_pcnt: float = 0.1):
        if not 0 <= masking_pcnt <= 1:
            raise ValueError("'masking_pcnt' should be in range [0, 1]!")
        if not 0 <= mask_with_random_word_pcnt <= 1:
            raise ValueError("'mask_with_random_word_pcnt' should be in range [0, 1]!")
        self.data: List[str] = texts
        self.tok: BasicTokenizer = tokenizer
        self.masking_prob: float = masking_pcnt
        self.mask_with_random_word_prob: float = mask_with_random_word_pcnt
        self.__mask_id = self.tok.mask_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[List[int], List[int]]:
        # TODO: add output for `token_type_ids`
        text = self.data[index]
        tokens = self.tok.tokenize(text)

        token_ids, mask = [self.__mask_id], []
        for tid in self.tok.convert_tokens_to_ids(tokens):
            if np.random.uniform() <= self.masking_prob:
                mask.append(tid)
                # random word instead of mask token
                if np.random.uniform() <= self.mask_with_random_word_prob:
                    masked_token_id = np.random.randint(0, self.tok.vocab_size)
                else:
                    masked_token_id = self.__mask_id
                token_ids.append(masked_token_id)
            else:
                mask.append(0)
                token_ids.append(tid)
        
        return token_ids, mask


class LinesDataset(TokensDataset):
    def __init__(self, file: str, *args, **kwargs):
        with open(file, "r") as f:
            texts = f.readlines()
        super().__init__(*args, texts=texts, **kwargs)


class MLMCollateFn:
    def __init__(self, 
                 pad_token_id: int,
                 mask_ignore_index: int, 
                 post_pad: bool = True):
        self.pad_value = pad_token_id
        self.ignore_index = mask_ignore_index
        self.post_pad = post_pad
        self.max_len = 512

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(tokens) for tokens, mask in batch)  # longes sequence in batch
        max_len = min(max_len, self.max_len)

        model_input = np.full(shape=(batch_size, max_len), fill_value=self.pad_value, dtype=int)
        mask_target = np.zeros(shape=(batch_size, max_len), dtype=int)

        for i, (tok_ids, mask_ids) in enumerate(batch):
            if self.post_pad:
                model_input[i, :len(tok_ids)] = np.array(tok_ids)[:max_len]
                mask_target[i, :len(mask_ids)] = np.array(mask_ids)[:max_len]
            else:
                model_input[i, -len(tok_ids):] = np.array(tok_ids)[:max_len]
                mask_target[i, -len(mask_ids):] = np.array(mask_ids)[:max_len]
        
        mask_target[mask_target == 0] = self.ignore_index
        
        model_input = torch.from_numpy(model_input)
        mask_target = torch.from_numpy(mask_target).view(-1)

        return {"input_ids": model_input, "mask": mask_target}
