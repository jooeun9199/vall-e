import torch
import torch.nn.functional as F
from torch import Tensor, nn
import random

def sin_emb(d_model, max_len=10000):
    
    theta = torch.arange(max_len/2)[:,None] / torch.pow(10000, torch.arange(0, d_model*2, 2) / d_model)
    emb = torch.cat((torch.sin(theta), torch.cos(theta)),dim=1).reshape(max_len,-1)

    return nn.Parameter(emb)


# class AdaLN(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.projection = nn.Linear(d_model,2)
#         self.layernorm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         """
#         Args:
#             x: (b t d)
#         """
#         h = self.layernorm(x)
#         s, b = self.projection(x).chunk(2,dim=-1) # (b t)
        
#         return s * h + b



class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, p_dropout):
        super().__init__()

        self.n_heads = n_heads
        self.n_layers = n_layers

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(p_dropout),
                )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, att_mask):
        
        b, t, t = att_mask.shape
        att_mask = att_mask.repeat(1,self.n_heads,1).reshape(b*self.n_heads,t,t).float()
        for _ in range(self.n_layers):
            x = self.norm1(x + self.dropout(self.attn(x, x, x, need_weights=False, attn_mask=att_mask)[0]))
            x = self.norm2(x + self.ffn(x))

        return x

class Emb(nn.Module):
    def __init__(self, n_codec, d_model, start_ind, end_ind, text_emb, wave_emb):
        super().__init__()
        self.n_codec = n_codec
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.text_emb = text_emb
        self.wave_emb = wave_emb
        self.pos_emb = sin_emb(d_model)

    def forward(self, text, prom, code=None, infer=False):

    # pre-process
        # ignored
        text[text == -1] = self.end_ind # (b t)
        prom[prom[...,0] == -1] = self.end_ind # (b t' L)
        code[code[...,0] == -1] = self.end_ind # (b t" L)
        # <s>
        text = F.pad(text, (1,0), 'constant', self.start_ind) # (b t+1)
        code = F.pad(code, (0,0,1,0), 'constant', self.start_ind) # (b t"+1 L)
        # </s>
        text = F.pad(text, (0,1), 'constant', self.end_ind) # (b t+2)
        code = F.pad(code, (0,0,0,1), 'constant', self.end_ind) # (b t"+2 L)
        gt = code

    # mask
        text_mask = F.pad((text != self.end_ind)[:,:-1], (1,0), 'constant', 1) # (b t+2)
        prom_mask = F.pad((prom != self.end_ind)[:,:-1,0], (1,0), 'constant', 1) # (b t')
        code_mask = F.pad((code != self.end_ind)[:,:-1,0], (1,0), 'constant', 1) # (b t"+2)
        mask = torch.cat((text_mask, prom_mask, code_mask), dim=1) # (b t+t'+t"+4)
        gt_mask = code_mask
        del text_mask, prom_mask, code_mask
        torch.cuda.empty_cache()
        mask = mask[:,:,None] * mask[:,None,:] # (b t+t'+t"+4 t+t'+t"+4)

    # embed
        # lookup
        text = self.text_emb(text) # (b t+2 d)
        prom = torch.stack([self.wave_emb[i](prom[...,i]) for i in range(self.n_codec)], dim=-1) # (b t' d L)
        code = torch.stack([self.wave_emb[i](code[...,i]) for i in range(self.n_codec)], dim=-1) # (b t"+2 d L)
        # positional encoding
        text += self.pos_emb[None,:text.shape[1],:] # (b t+2 d)
        prom += self.pos_emb[None,:prom.shape[1],:,None] # (b t' d L)
        code += self.pos_emb[None,:code.shape[1],:,None] # (b t"+2 d L)

        return text, prom, code, mask, gt, gt_mask


class AR(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, p_dropout, ignore_ind, wave_emb):
        super().__init__()
        self.transformer = Transformer(d_model, n_heads, n_layers, p_dropout)
        self.ignore_ind = ignore_ind
        self.wave_emb = wave_emb

    def forward(self, text, prom, code, mask, gt, gt_mask, infer=False, sampling_temperature=1.0) -> Tensor:
        """
        Args:
            text: (b t+2 d), prompt text tokens
            prom: (b t' d), prompt acoustic tokens of level 0
            code: (b t"+2 d), gt acoustic tokens of level 0
            mask: (b t+t'+t"+4 t+t'+t"+4)
            gt: (b t"+2)
            gt_mask: (b t"+2)
            infer: train mode if False, inference mode if True
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            l: loss of AR model
        """
        # update mask
        mask[:,:code.shape[1],-code.shape[1]:] = 0
        mask[:,-code.shape[1]:,-code.shape[1]:] = torch.tril(mask[:,-code.shape[1]:,-code.shape[1]:])
        
        # update input
        x = torch.cat((text,prom,code),dim=1) # (b t+t'+t"+4 d)

        # transformer
        x = self.transformer(x, mask) # (b t+t'+t"+4 d)

        # classifier
        h = x @ self.wave_emb.weight.t() # (b t+t'+t"+4 n_vocab)

        # loss
        h = h[:,-gt.shape[1]-1:-1,:] * gt_mask[...,None]
        y = gt * gt_mask + self.ignore_ind * ~gt_mask
        l = F.cross_entropy(h.permute(0,2,1), y, ignore_index=self.ignore_ind)

        return l


class NAR(nn.Module):
    def __init__(self, n_codec, d_model, n_heads, n_layers, p_dropout, ignore_ind, wave_emb):
        super().__init__()
        self.n_codec = n_codec
        self.transformer = Transformer(d_model, n_heads, n_layers, p_dropout)
        self.ignore_ind = ignore_ind
        self.wave_emb = wave_emb

    def forward(self, text, prom, code, mask, gt, gt_mask, infer=False, sampling_temperature=1.0) -> Tensor:
        """
        Args:
            text: (b t+2 d), prompt text tokens
            prom: (b t' d L), prompt acoustic tokens
            code: (b t"+2 d L), gt acoustic tokens
            mask: (b t+t'+t"+4 t+t'+t"+4)
            gt: (b t"+2 L)
            gt_mask: (b t"+2)
            infer: train mode if False, inference mode if True
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            l: loss of NAR model
        """
        
        i = random.randrange(1, self.n_codec)

        # update input
        prom = torch.sum(prom, dim=-1)
        code = torch.sum(code[...,:i], dim=-1)
        x = torch.cat((text,prom,code),dim=1) # (b t+t'+t"+4 d)

        # transformer
        x = self.transformer(x, mask) # (b t+t'+t"+4 d)

        # classifier
        h = x @ self.wave_emb[i].weight.t() # (b t+t'+t"+4 n_vocab)

        # loss
        h = h[:,-gt.shape[1]:,:] * gt_mask[...,None]
        y = gt[...,i] * gt_mask + self.ignore_ind * ~gt_mask
        l = F.cross_entropy(h.permute(0,2,1), y, ignore_index=self.ignore_ind)

        return l
    

class VallE(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_codec: int = 8,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        p_dropout: float = 0.1,
    ):
        """
        Args:
            n_vocab: size of vocab
            n_codec: number of codebooks
            d_model: size of embedding dimension
            n_heads: number of transformer heads
            n_layers: number of transformer iterations
            p_dropout: dropout
        """
        super().__init__()
        start_ind = d_model
        end_ind = d_model + 1
        ignore_ind = -1
        text_emb = nn.Embedding(n_vocab + 2, d_model)  # tokens for <s> and </s>
        wave_emb = nn.ModuleList(nn.Embedding(n_vocab + 2, d_model) for _ in range(n_codec)) # tokens for <s> and </s>
        
        self.emb = Emb(n_codec, d_model, start_ind, end_ind, text_emb, wave_emb)
        self.AR = AR(d_model, n_heads, n_layers, p_dropout, ignore_ind, wave_emb[0])
        self.NAR = NAR(n_codec, d_model, n_heads, n_layers, p_dropout, ignore_ind, wave_emb)

    def forward(self, text, prom, code=None, infer=False, sampling_temperature=1.0) -> Tensor:
        """
        Args:
            text: (b t), prompt text tokens
            prom: (b t' L), prompt acoustic tokens
            code: (b t" L), gt acoustic tokens
            infer: train mode if False, inference mode if True
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            [l_AR, l_NAR]: each loss of AR, NAR model
        """
        if not infer:
            assert code is not None, 'need ground truth input for training the model'

        # embed inputs
        text, prom, code, mask, gt, gt_mask = self.emb(text, prom, code, infer)

        # model
        l_AR = self.AR(text, prom[...,0], code[...,0], mask, gt[...,0], gt_mask, infer, sampling_temperature)
        l_NAR = self.NAR(text, prom, code, mask, gt, gt_mask, infer, sampling_temperature)

        return [l_AR, l_NAR]

# y = torch.distributions.Categorical(logits=h / sampling_temperature).sample()