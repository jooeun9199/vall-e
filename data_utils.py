import torch
from torch.utils.data import Dataset
from torch import tensor, Tensor
from collections import defaultdict 
import math
import random


class VALLEDataset(Dataset):
    def __init__(self, paths, sr, prompt_s_len):
        self.paths = paths
        self.txt_ext = '.phn.txt'
        self.wav_ext = '.qnt.pt'
        self.sr = sr
        self.prompt_s_len = prompt_s_len
        self.prompt_t_len = self._get_prompt_t_len()
        self.phones = sorted(set().union(*[self._get_phones(path) for path in self.paths]))
        self.phone_map = {phone: i for i, phone in enumerate(self.phones, 1)}
        self.paths_by_spkr = self._get_paths_by_spkr()
        self._validate_path()

    def __getitem__(self, index):
        path = self.paths[index]

        text = tensor(list(map(self.phone_map.get, self._get_phones(path))))
        prom = self._get_prom(path)
        code = self._get_code(path)

        return text, prom, code

    def __len__(self):
        return len(self.paths)

    def _get_phones(self, path) -> list:
        path += self.txt_ext
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
            phones = ["<s>"] + content.split() + ["</s>"]
            return phones

    def _get_prom(self, path) -> Tensor:
        """
        Returns:
            prom: (t q)
        """
        spkr = path.split('_')[0]
        choices = list(set(self.paths_by_spkr[spkr]) - {path})
        choice = random.choice(choices) + self.wav_ext
        lst = [torch.load(choice)[0].t()]
        while torch.cat(lst).shape[0] < self.prompt_t_len:
            choice = random.choice(choices) + self.wav_ext
            lst.append(torch.load(choice)[0].t())
        return torch.cat(lst)[:self.prompt_t_len,]

    def _get_code(self, path) -> Tensor:
        """
        Returns:
            code: (t q)
        """
        path += self.wav_ext
        code = torch.load(path)[0].t()

        return code

    def _get_paths_by_spkr(self) -> dict:
        paths_by_spkr = defaultdict(list)

        for path in self.paths:
            paths_by_spkr[path.split('_')[0]].append(path)

        return paths_by_spkr

    def _get_prompt_t_len(self) -> int:
        strides = [8,5,4,2]
        ret = self.sr * self.prompt_s_len
        for stride in strides:
            ret = math.ceil(ret/stride)
            
        return ret

    def _validate_path(self):
        for k,lst in self.paths_by_spkr.copy().items():
            if len(lst) == 1:
                self.paths_by_spkr.pop(k)
                self.paths.remove(lst[0])



def collate_fn(batch):
    """
    Args:
        batch: [(text, prom, code) * b]
    Returns:
        padded batches and lengths for each batches
        text_batch: (b t), prompt text tokens
        text_len:   (b), prompt text lengths
        prom_batch: (b t' l), prompt acoustic tokens
        prom_len:   (b), prompt acoustic lengths
        code_batch: (b t" l), gt acoustic tokens
        code_len:   (b), gt acoustic lengths
    """
    text_max_len, prom_max_len, code_max_len = torch.max(torch.stack([tensor([X.shape[0] for X in Xs]) for Xs in batch]), dim=0)[0]
    batch_size = len(batch)
    codec_level = batch[0][1].shape[-1]

    text_batch = torch.full((batch_size, text_max_len), -1)
    prom_batch = torch.full((batch_size, prom_max_len, codec_level), -1)
    code_batch = torch.full((batch_size, code_max_len, codec_level), -1)
    # text_len = torch.empty(batch_size)
    # prom_len = torch.empty(batch_size)
    # code_len = torch.empty(batch_size)
    for i, (text, prom, code) in enumerate(batch):
        # tl = text_len[i] = text.shape[0]
        # pl = prom_len[i] = prom.shape[0]
        # cl = code_len[i] = code.shape[0]
        text_batch[i,:text.shape[0]] = text
        prom_batch[i,:prom.shape[0],:] = prom
        code_batch[i,:code.shape[0],:] = code
    
    return text_batch, prom_batch, code_batch
