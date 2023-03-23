import torch
import utils
from vall_e.vall_e.ar import AR
from vall_e.vall_e.nar import NAR


ckpt_dir_AR = './ckpts/AR'
ckpt_dir_NAR = './ckpts/NAR'
out_dir = './outs/infer'
ckpt_num_AR = '*' # for latest: type '*'
ckpt_num_NAR = '*' # for latest: type '*'
num_vocab = 1024
sr = 24000
lr = 1e-5

model_AR = AR(num_vocab)
model_NAR = NAR(num_vocab)
optimizer_AR = torch.optim.AdamW(model_AR.parameters(), lr=lr)
optimizer_NAR = torch.optim.AdamW(model_NAR.parameters(), lr=lr)

with torch.no_grad():

    if ckpt_num_AR == '*':
        AR_epoch = utils.load_checkpoint(utils.latest_checkpoint_path(ckpt_dir_AR, "G_*.pth"), model_AR, optimizer_AR)
    else:
        AR_epoch = utils.load_checkpoint(os.path.join(ckpt_dir_AR, "G_"+ckpt_num_AR+".pth"), model_AR, optimizer_AR)
    
    if ckpt_num_NAR == '*':
        NAR_epoch = utils.load_checkpoint(utils.latest_checkpoint_path(ckpt_dir_NAR, "G_*.pth"), model_NAR, optimizer_NAR)
    else:
        NAR_epoch = utils.load_checkpoint(os.path.join(ckpt_dir_NAR, "G_"+ckpt_num_NAR+".pth"), model_NAR, optimizer_NAR)
    

    code = model_AR([text_list[-1].to(device)], [prom_list[-1].to(device)])[0].unsqueeze(-1) # (t 1)

    for i in range(1, 8):
        code = model([text_list[-1].to(device)], [prom_list[-1].to(device)], [code[...,:i].to(device)]) # [(t L) * 1]
        code = code[0] # (t L)

    decode_model = EncodecModel.encodec_model_24khz()
    decode_model.set_target_bandwidth(6.0)
    wave = decode_model.decode([(code.t().unsqueeze(0).cpu(), None)])

soundfile.write("{}/AR{}_NAR{}.wav".format(out_dir, AR_epoch, NAR_epoch), wave[0, 0], sr)