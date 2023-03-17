import torch
from torch.utils.data import Dataset
from torch import tensor, Tensor
from collections import defaultdict 
import math
import random


class VallEDataset(Dataset):
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



def collate_fn(data_list):
    """
    Args:
        data_list: [(text, prom, code) * b]
    Returns:
        text_list: [(t) * b], prompt text tokens
        prom_list: [(t' L) * b], prompt acoustic tokens
        code_list: [(t" L) * b], gt acoustic tokens
    """
    text_list, prom_list, code_list = [[x[j] for x in data_list] for j in range(len(data_list[0]))]
    
    return text_list, prom_list, code_list
