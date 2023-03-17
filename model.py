import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn
from einops import rearrange
import random
import gc


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

    def forward(self, x, m):
    
        # single batch
        if x.shape[0] == 1:
            m = m[0]
            x = x[0]
        else:
            b, t, t = m.shape
            m = m.repeat(1,self.n_heads,1).reshape(b*self.n_heads,t,t)
    
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(self.n_layers):
            x = self.norm1(x + self.dropout(self.attn(x, x, x, need_weights=False, attn_mask=m)[0]))
            x = self.norm2(x + self.ffn(x))

        if len(x.shape)==2:
            x = x.unsqueeze(0)
        
        return x

class Emb(nn.Module):
    def __init__(self, n_codec, d_model, sep_emb, text_emb, wave_emb):
        super().__init__()
        self.n_codec = n_codec
        self.pos_emb = self.sin_emb(d_model)
        self.sep = sep_emb
        self.text_emb = text_emb
        self.wave_emb = wave_emb

    def sin_emb(self, d_model, max_len=10000):
        theta = torch.arange(max_len/2)[:,None] / torch.pow(10000, torch.arange(0, d_model*2, 2) / d_model)
        emb = torch.cat((torch.sin(theta), torch.cos(theta)),dim=1).reshape(max_len,-1)

        return nn.Parameter(emb)

    def forward(self, text, prom, code=None, level=None):
        """
        Args:
            text: [(t) * b], prompt text tokens
            prom: [(t' L) * b], prompt acoustic tokens
            code: [(t" L) * b], gt acoustic tokens
            level: level to be optimized in NAR, only for train mode (1~7)
        Returns:
            x: (b T d)
                text <sep> prom0 <sep> code0    AR train  if level==0 
                text <sep> promN <sep> codei    NAR train if level==i 
                text <sep> prom0 <sep>          AR infer  if code is None 
                text <sep> promN <sep> code0    NAR infer if code has single level 
            m: (b T 1), mask for padded x

        """
        device = self.text_emb.weight.device
        text = self.text_emb(torch.cat(text).to(device)).split([*map(len, text)])

        # AR train
        if level==0:
            prom = self.wave_emb[0](torch.cat(prom)[...,0].to(device)).split([*map(len, prom)])
            code = self.wave_emb[0](torch.cat(code)[...,0].to(device)).split([*map(len, code)])
            x = [torch.cat((text, self.sep, prom, self.sep, code)) for text, prom, code in zip(text, prom, code)]
        
        # NAR train
        elif level is not None:
            prom = sum([self.wave_emb[i](torch.cat(prom)[...,i].to(device)) for i in range(self.n_codec)]).split([*map(len, prom)])
            code = sum([self.wave_emb[i](torch.cat(code)[...,i].to(device)) for i in range(level+1)]).split([*map(len, code)])
            x = [torch.cat((text, self.sep, prom, self.sep, code)) for text, prom, code in zip(text, prom, code)]

        # AR infer
        elif code is None:
            prom = self.wave_emb[0](torch.cat(prom)[...,0].to(device)).split([*map(len, prom)])
            x = [torch.cat((text, self.sep, prom, self.sep)) for text, prom in zip(text, prom)]
        
        # NAR infer
        elif code[0].shape[1]==1:
            prom = sum([self.wave_emb[i](torch.cat(prom)[...,i].to(device)) for i in range(self.n_codec)]).split([*map(len, prom)])
            code = self.wave_emb[0](torch.cat(code)[...,0].to(device)).split([*map(len, code)])
            x = [torch.cat((text, self.sep, prom, self.sep, code)) for text, prom, code in zip(text, prom, code)]
        
        else:
            raise ValueError
            
        l = torch.tensor(list(map(len, x))) # (b)
        x = rearrange(pad_sequence(x), "t b d -> b t d")
        x = x + self.pos_emb[None,:x.shape[1]]

        return x, l


class AR(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, p_dropout, eos_ind, sep_emb, text_emb, wave_emb):
        super().__init__()
        self.transformer = Transformer(d_model, n_heads, n_layers, p_dropout)
        self.eos_ind = eos_ind
        self.sep_emb = sep_emb
        self.text_emb = text_emb
        self.wave_emb = wave_emb

    def forward(self, x, l, sampling_temperature, text=None, code=None, max_ar_step=None) -> Tensor:
        """
        Args:
            x: (b T d)
            l: (b), lengths of x
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
            text: [(t) * b], prompt text tokens
            code: [(t" L) * b], gt acoustic tokens
            max_ar_step: max generated token length
        Returns:
            loss: loss of AR model if train mode
            out: generated AR tokens if infer mode
        """
        
        # mask
        m = (torch.arange(max(l))[None] < l[:,None]).to(x) # (b T)
        m = m.unsqueeze(1) * m.unsqueeze(2) # (b T T)
        m = m.tril()

        # train
        if None not in (text, code):

            # transformer
            x = self.transformer(x, m) # (b T d)

            # classifier
            h_text = torch.cat([x[:len(text[i])-1] @ self.text_emb.weight.t() for i, x in enumerate(x)]) # (T N)
            h_code = torch.cat([x[l[i]-len(code[i])-1:l[i]] @ self.wave_emb.weight.t() for i, x in enumerate(x)]) # (T' N+1)

            # loss
            y_text = torch.cat([text[1:].to(x.device) for text in text])
            y_code = torch.cat([torch.cat((code[:,0], torch.tensor([self.eos_ind]))).to(x.device) for code in code])

            loss = F.cross_entropy(h_text, y_text) + F.cross_entropy(h_code, y_code)

            return loss

        # infer
        elif max_ar_step is not None:

            out = []

            for i in range(max_ar_step):

                # transformer
                h = self.transformer(x, m) # (1 T d)

                # classifier
                h = h[:,-1] @ self.wave_emb.weight.t() # (1 1 n_vocab)

                # generate next token
                y = torch.distributions.Categorical(logits=h / sampling_temperature).sample()

                # update
                out.append(y)
                x = torch.cat((x, self.wave_emb(y)[None]), dim=1) # (1 T d)
                m = F.pad(m, (0,1), 'constant', 0) # (1 T T+1)
                m = F.pad(m, (0,0,0,1), 'constant', 1) # (1 T+1 T+1)
                
                # end
                if y == self.eos_ind:
                    break

            return [torch.tensor(out).unsqueeze(-1).to(x.device)] # [(t" 1) * 1]
        
        else:
            raise ValueError



class NAR(nn.Module):
    def __init__(self, n_codec, d_model, n_heads, n_layers, p_dropout, start_ind, end_ind, ignore_ind, wave_emb):
        super().__init__()
        self.n_codec = n_codec
        self.transformer = Transformer(d_model, n_heads, n_layers, p_dropout)
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.ignore_ind = ignore_ind
        self.wave_emb = wave_emb

    def forward(self, x, l, sampling_temperature, code=None, level=None) -> Tensor:
        """
        Args:
            x: (b T d)
            l: (b), lengths of x
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
            code: [(t" L) * b], gt acoustic tokens if train previous tokens if infer
            level: codec level to optimize
        Returns:
            loss: loss of NAR model if train mode
            out: generated NAR tokens if infer mode
        """

        # mask
        m = (torch.arange(max(l))[None] < l[:,None]).to(x) # (b T)
        m = m.unsqueeze(1) * m.unsqueeze(2) # (b T T)

        # train
        if level is not None:

            # transformer
            x = self.transformer(x, m) # (b T d)

            # classifier
            h = torch.cat([x[l[i]-len(code[i]):l[i]] @ self.wave_emb[level].weight.t() for i, x in enumerate(x)]) # (b T n_vocab)

            # loss
            y = torch.cat([code[:,level].to(x.device) for code in code])
            loss = F.cross_entropy(h, y)

            return loss

        # infer
        elif code is not None:
            
            code = code[0][None] # (1 t L)

            for i in range(1,self.n_codec):

                # transformer
                h = self.transformer(x, m) # (b T d)

                # classifier
                h = h[:,-code.shape[1]:] @ self.wave_emb[i].weight.t() # (b t n_vocab)

                # generate next token
                y = torch.distributions.Categorical(logits=h / sampling_temperature).sample().to(x.device) # (b t)

                # update
                code = torch.cat((code, y.unsqueeze(-1)), dim=-1)
                x[:,-code.shape[1]:] += self.wave_emb[i](y) # (b t d)
                
            return [code[0]] # [(t L) * 1]

        else:
            raise ValueError

    

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
        """
        super().__init__()
        eos_ind = d_model
        ignore_ind = -1
        sep_emb = nn.Parameter(torch.randn(d_model).unsqueeze(0))
        text_emb = nn.Embedding(n_vocab, d_model)
        wave_emb = nn.ModuleList([nn.Embedding(n_vocab + 1, d_model)] + [nn.Embedding(n_vocab, d_model) for _ in range(n_codec)]) # <EOS> token
        self.n_codec = n_codec

        self.emb = Emb(n_codec, d_model, sep_emb, text_emb, wave_emb)
        self.AR = AR(d_model, n_heads, n_layers, p_dropout, eos_ind, sep_emb, text_emb, wave_emb[0])
        self.NAR = NAR(n_codec, d_model, n_heads, n_layers, p_dropout, eos_ind, sep_emb, ignore_ind, wave_emb)

    def forward(self, text, prom, code=None, infer=False, sampling_temperature=1.0, max_ar_step=300) -> Tensor:
        """
        Args:
            text: [(t) * b], prompt text tokens
            prom: [(t' L) * b], prompt acoustic tokens
            code: [(t" L) * b], gt acoustic tokens
            infer: train mode if False, inference mode if True
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            [l_AR, l_NAR]: each loss of AR, NAR model
        """
        if infer:
            assert len(text) == 1 and len(prom) == 1, 'allow non-batch single data only'
        else:
            assert code is not None, 'need ground truth input for training the model'

        # generate
        if infer:

            x, l = self.emb(text, prom)
            
            y = self.AR(x, l, sampling_temperature, max_ar_step=max_ar_step)

            x, l = self.emb(text, prom, y)

            y = self.NAR(x, l, sampling_temperature, y)

            return y
        
        # losses
        else:

            x, l = self.emb(text, prom, code, 0)
            
            # import pdb; pdb.set_trace()
            l_AR = self.AR(x, l, sampling_temperature, text, code)

            # random level (1~6) for NAR
            i = random.randrange(1, self.n_codec-1)

            x, l = self.emb(text, prom, code, i)

            l_NAR = self.NAR(x, l, sampling_temperature, code, i+1)

            return [l_AR, l_NAR]


        