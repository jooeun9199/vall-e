import torch
import torch.nn.functional as F
from torch import Tensor, nn

def sin_emb(d_model, max_len=10000):
    
    theta = torch.arange(max_len/2)[:,None] / torch.pow(10000, torch.arange(0, d_model*2, 2) / d_model)
    emb = torch.cat((torch.sin(theta), torch.cos(theta)),dim=1).reshape(max_len,-1)

    return nn.Parameter(emb)


class AdaLN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model,2)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (b t d)
        """
        h = self.layernorm(x)
        s, b = self.projection(x).chunk(2,dim=-1) # (b t)
        
        return s * h + b



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


class AR(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        p_dropout: float = 0.1,
        n_codec: int = 8
    ):
        """
        Args:
            n_vocab: size of vocab
            n_codec: codec levels
        """
        super().__init__()
        self.ignore_ind = -1
        self.d_model = d_model
        self.start_ind = d_model
        self.end_ind = d_model + 1
        self.text_emb = nn.Embedding(n_vocab + 2, d_model)  # tokens for <s> and </s>
        self.wave_emb = nn.ModuleList(nn.Embedding(n_vocab + 2, d_model) for _ in range(n_codec)) # tokens for <s> and </s>
        self.pos_emb = sin_emb(d_model)
        self.transformer = Transformer(d_model, n_heads, n_layers, p_dropout)
        

    def forward(self, text_batch, prom_batch, code_batch=None, infer=False, sampling_temperature=1.0) -> Tensor:
        """
        Args:
            text_batch: (b t), prompt text tokens
            prom_batch: (b t' l), prompt acoustic tokens
            code_batch: (b t" l), gt acoustic tokens
            infer: train mode if False, inference mode if True
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            y: sampled tokens
        """
        if not infer:
            assert code_batch is not None, 'need ground truth input for training the model'

    # embed
        # ignored
        text_batch[text_batch == -1] = self.end_ind # (b t)
        prom_batch[prom_batch[...,0] == -1] = self.end_ind # (b t' l)
        code_batch[code_batch[...,0] == -1] = self.end_ind # (b t" l)
        # <s>
        text_batch = F.pad(text_batch, (1,0), 'constant', self.start_ind) # (b t+1)
        code_batch = F.pad(code_batch, (0,0,1,0), 'constant', self.start_ind) # (b t"+1 l)
        # </s>
        text_batch = F.pad(text_batch, (0,1), 'constant', self.end_ind) # (b t+2)
        code_batch = F.pad(code_batch, (0,0,0,1), 'constant', self.end_ind) # (b t"+2 l)
        # embedding lookup
        text_emb = self.text_emb(text_batch)           # (b t+2 d)
        prom_emb = self.wave_emb[0](prom_batch[...,0]) # (b t' d)
        code_emb = self.wave_emb[0](code_batch[...,0]) # (b t"+2 d)
        # positional encoding
        text_emb += self.pos_emb[None,:text_batch.shape[1],:] # (b t+2 d)
        prom_emb += self.pos_emb[None,:prom_batch.shape[1],:] # (b t' d)
        code_emb += self.pos_emb[None,:code_batch.shape[1],:] # (b t"+2 d)
        # concat
        x = torch.cat((text_emb, prom_emb, code_emb), dim=1) # (b t+t'+t"+4 d)
    
    # transformer
        text_mask = F.pad((text_batch != self.end_ind)[:,:-1], (1,0), 'constant', 1) # (b t+2)
        prom_mask = F.pad((prom_batch != self.end_ind)[:,:-1,0], (1,0), 'constant', 1) # (b t')
        code_mask = F.pad((code_batch != self.end_ind)[:,:-1,0], (1,0), 'constant', 1) # (b t"+2)
        mask = torch.cat((text_mask, prom_mask, code_mask), dim=1) # (b t+t'+t"+4)
        att_mask = mask[:,:,None] * mask[:,None,:] # (b t+t'+t"+4 t+t'+t"+4)
        att_mask[:,:code_batch.shape[1],-code_batch.shape[1]:] = 0
        att_mask[:,-code_batch.shape[1]:,-code_batch.shape[1]:] = torch.tril(att_mask[:,-code_batch.shape[1]:,-code_batch.shape[1]:])
        x = self.transformer(x, att_mask) # (b t+t'+t"+4 d)

    # classifier
        h = x @ self.wave_emb[0].weight.t() # (b t+t'+t"+4 n_vocab)

    # loss
        h = h[:,-code_batch.shape[1]-1:-1,:] * code_mask[...,None]
        y = code_batch[...,0] * code_mask + self.ignore_ind * ~code_mask
        l = F.cross_entropy(h.permute(0,2,1), y, ignore_index=self.ignore_ind)

    # output
        y = torch.distributions.Categorical(logits=h / sampling_temperature).sample()
        return y, l