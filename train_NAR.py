import os
import math
import random
import soundfile
from encodec import EncodecModel

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import deepspeed

import utils
from data_utils import VallEDataset, collate_fn
from vall_e.vall_e.ar import AR
from vall_e.vall_e.nar import NAR


# config_dir = './configs/base.json'
name = 'NAR'
name_AR = 'AR'
data_dir = './data/LibriTTS/'
log_dir = './logs/' + name
ckpt_dir = './ckpts/' + name
ckpt_dir_AR = './ckpts/' + name_AR
out_dir = './outs/' + name
ckpt_num = '*' # for latest: type '*'
global_step = 0
total_steps = 80000000
devices = [2,3]
num_vocab = 1024
sr = 24000
prompt_s_len = 3
lr = 1e-5
warmup_num_steps = 32000
split_ratio = 0.95
num_workers = 4
batch_size = 2
log_interval = 20


def main():

    n_gpus = len(devices)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12124'

    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus,))


def train_and_eval(rank, n_gpus):

    # GPU
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    # Model
    model = NAR(num_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = deepspeed.runtime.lr_schedules.WarmupDecayLR(optimizer, total_num_steps=total_steps, warmup_num_steps= warmup_num_steps)
    
    model_AR = AR(num_vocab).to(device)
    optimizer_AR = torch.optim.AdamW(model_AR.parameters(), lr=lr)

    # Data
    paths = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.qnt.pt'):
            paths.append(data_dir + filename.split('.')[0])
        
    random.seed(0)
    random.shuffle(paths)
    N = round(len(paths) * split_ratio)
    train_paths = paths[:N]
    val_paths = paths[N:]

    train_dataset = VallEDataset(train_paths, sr, prompt_s_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=batch_size, drop_last=True, sampler=train_sampler)
    total_epochs = math.ceil(total_steps / len(train_loader))

    if rank == 0:
        val_dataset = VallEDataset(val_paths, sr, prompt_s_len)
        val_loader = DataLoader(val_dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=batch_size)

    # Load
    try:
        if ckpt_num == '*':
            epoch = utils.load_checkpoint(utils.latest_checkpoint_path(ckpt_dir, "G_*.pth"), model, optimizer)
        else:
            epoch = utils.load_checkpoint(os.path.join(ckpt_dir, "G_"+ckpt_num+".pth"), model, optimizer)
        optimizer.step_num = epoch * len(train_loader)
        global_step = epoch * len(train_loader)
        epoch_str = epoch + 1

    except:
        epoch_str = 1
        global_step = 0
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


    # Logger
    if rank == 0:
        logger = utils.get_logger(log_dir)

    # Train
    for epoch in range(epoch_str, total_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        
        model.train()

        for batch_idx, (text_list, prom_list, code_list) in enumerate(train_loader):

            text_list, prom_list, code_list = map(lambda x_list: [x.to(device) for x in x_list], [text_list, prom_list, code_list])

            optimizer.zero_grad()
            _ = model(text_list, prom_list, code_list)

            loss = sum(model.loss.values())
            loss.backward()

            optimizer.step()

            scheduler.step()

            if rank == 0:

                if batch_idx % log_interval == 0:

                    logger.info('Train Epoch: {}, Global Step: {} [{}/{} ({:.0f}%)]   \tLoss: {:.6f}'.format(
                        epoch,
                        global_step,
                        batch_idx * batch_size,
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss))

            global_step += 1

        if rank == 0:
            
            # Eval
            logger.info('============= Epoch: {} ============='.format(epoch))

            model.eval()

            loss_sum = 0
            with torch.no_grad():
                for batch_idx, (text_list, prom_list, code_list) in enumerate(val_loader):

                    text_list, prom_list, code_list = map(lambda x_list: [x.to(device) for x in x_list], [text_list, prom_list, code_list])

                    _ = model(text_list, prom_list, code_list)
                    
                    loss = sum(model.loss.values())
                    loss_sum += loss
                            
                    if rank == 0:

                        if batch_idx % log_interval == 0:

                            logger.info('Eval: [{}/{} ({:.0f}%)]   \tLoss: {:.6f}'.format(
                                batch_idx * batch_size,
                                len(val_loader.dataset),
                                100. * batch_idx / len(val_loader),
                                loss))

            loss = loss_sum / len(val_loader.dataset)
            logger.info('Average Loss for {} Eval data: {:.6f}'.format(len(val_loader.dataset), loss))

            # Infer
            with torch.no_grad():

                AR_epoch = utils.load_checkpoint(utils.latest_checkpoint_path(ckpt_dir_AR, "G_*.pth"), model_AR, optimizer_AR)

                code = model_AR([text_list[-1].to(device)], [prom_list[-1].to(device)])[0].unsqueeze(-1) # (t 1)

                for i in range(1, 8):
                    code = model([text_list[-1].to(device)], [prom_list[-1].to(device)], [code[...,:i].to(device)]) # [(t L) * 1]
                    code = code[0] # (t L)

                decode_model = EncodecModel.encodec_model_24khz()
                decode_model.set_target_bandwidth(6.0)
                wave = decode_model.decode([(code.t().unsqueeze(0).cpu(), None)])

            soundfile.write("{}/AR{}_NAR{}.wav".format(out_dir, AR_epoch, epoch), wave[0, 0], sr)

            # Save
            utils.save_checkpoint(model, optimizer, lr, epoch, os.path.join(ckpt_dir, "G_{}.pth".format(epoch)))
            logger.info('======================================'.format(epoch))

if __name__ == "__main__":
    main()

