import os
import math
import random
import soundfile
from einops import rearrange
from encodec import EncodecModel

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import deepspeed

import utils
from data_utils import VallEDataset, collate_fn
from model import VallE


# config_dir = './configs/base.json'
data_dir = './data/LibriTTS/'
log_dir = './logs/base/'
ckpt_dir = './ckpts/base/'
out_dir = './outs/base/'
ckpt_num = '*' # for latest: type '*'
global_step = 0
total_steps = 800000
devices = [2,3,4,5]
num_vocab = 1024
sr = 24000
prompt_s_len = 3
lr = 1e-5
warmup_num_steps = 32000
split_ratio = 0.95
num_workers = 4
batch_size = 1
log_interval = 20


def main():

    n_gpus = len(devices)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13451'

    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus,))


def train_and_eval(rank, n_gpus):

    # GPU
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    # Model
    model = VallE(num_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = deepspeed.runtime.lr_schedules.WarmupDecayLR(optimizer, total_num_steps=total_steps, warmup_num_steps= warmup_num_steps)
    
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

            optimizer.zero_grad()
            losses = model(text_list, prom_list, code_list, infer=False)

            loss = sum(losses)

            loss.backward()

            optimizer.step()

            scheduler.step()

            if rank == 0:

                if batch_idx % log_interval == 0:

                    logger.info('Train Epoch: {}, Global Step: {} [{}/{} ({:.0f}%)]   \tTotal Loss: {:.6f}, AR: {:.6f}, NAR: {:.6f}'.format(
                        epoch,
                        global_step,
                        batch_idx * batch_size,
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss, losses[0], losses[1]))

            global_step += 1

        if rank == 0:
            
            # Eval
            logger.info('============= Epoch: {} ============='.format(epoch))

            model.eval()

            loss_sum = 0
            with torch.no_grad():
                for batch_idx, (text_list, prom_list, code_list) in enumerate(val_loader):
                    
                    [(a.to(device), b.to(device), c.to(device)) for a,b,c in zip(text_list,prom_list,code_list)]

                    losses = model(text_list, prom_list, code_list, infer=False)
                    
                    loss = sum(losses)

                    loss_sum += loss

                            
                    if rank == 0:

                        if batch_idx % log_interval == 0:

                            logger.info('Eval: [{}/{} ({:.0f}%)]   \tTotal Loss: {:.6f}, AR: {:.6f}, NAR: {:.6f}'.format(
                                batch_idx * batch_size,
                                len(val_loader.dataset),
                                100. * batch_idx / len(val_loader),
                                loss, losses[0], losses[1]))    
                    
            
            loss = loss_sum / len(val_loader.dataset)
            logger.info('Average Loss for {} Eval data: {:.6f}'.format(len(val_loader.dataset), loss))

            # Infer
            with torch.no_grad():
                code = model([text_list[-1]], [prom_list[-1]], infer=True, sampling_temperature=0.2) # [(t L) * 1]
                code = code[0].T.unsqueeze(0)

                decode_model = EncodecModel.encodec_model_24khz()
                decode_model.set_target_bandwidth(6.0)
                wave = decode_model.decode([(code.cpu(), None)])
                            
            soundfile.write("{}/{}.wav".format(out_dir, epoch), wave[0, 0], sr)

            # Save
            utils.save_checkpoint(model, optimizer, lr, epoch, os.path.join(ckpt_dir, "G_{}.pth".format(epoch)))
            logger.info('======================================'.format(epoch))

if __name__ == "__main__":
    main()

