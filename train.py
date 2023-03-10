import os
import math
import random
import soundfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import deepspeed

import utils
from data_utils import VALLEDataset, collate_fn
from model import AR
from vall_e.emb.qnt import decode

# config_dir = './configs/AR.json'
data_dir = './data/LibriTTS/'
log_dir = './logs/AR/'
ckpt_dir = './ckpts/AR/'
ckpt_num = '*' # for latest: type '*'
global_step = 0
total_steps = 800000
devices = [4,5,6,7]
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
    os.environ['MASTER_PORT'] = '12345'

    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus,))


def train_and_eval(rank, n_gpus):

    # GPU
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    # Model
    model = AR(num_vocab).to(device)
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

    train_dataset = VALLEDataset(train_paths, sr, prompt_s_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=batch_size, drop_last=True, sampler=train_sampler)
    total_epochs = math.ceil(total_steps / len(train_loader))

    if rank == 0:
        val_dataset = VALLEDataset(val_paths, sr, prompt_s_len)
        val_loader = DataLoader(val_dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=batch_size)

    # Load
    try:
        if ckpt_num == '*':
            epoch = utils.load_checkpoint(utils.latest_checkpoint_path(ckpt_dir, "G_*.pth"), model, optimizer)
        else:
            epoch = utils.load_checkpoint(os.path.join(ckpt_dir, "G_"+ckpt_num+".pth"), model, optimizer)
        optimizer.step_num = epoch * len(train_loader)
        optimizer._update_learning_rate()
        global_step = epoch * len(train_loader)
        epoch_str = epoch + 1

    except:
        epoch_str = 1
        global_step = 0
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)


    # Logger
    if rank == 0:
        logger = utils.get_logger(log_dir)

    end = False
    # Train
    for epoch in range(epoch_str, total_epochs + 1):

        train_loader.sampler.set_epoch(epoch)
        
        model.train()

        for batch_idx, (text_batch, prom_batch, code_batch) in enumerate(train_loader):
            text_batch, prom_batch, code_batch =  map(lambda t: t.to(device), [text_batch, prom_batch, code_batch])
            
            optimizer.zero_grad()
            out, loss = model(text_batch, prom_batch, code_batch, infer=False)

            loss.backward()

            optimizer.step()

            scheduler.step()

            if rank == 0:

                if batch_idx % log_interval == 0:

                    logger.info('Train Epoch: {}, Global Step: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch,
                        global_step,
                        batch_idx * batch_size,
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader)))
                    logger.info('Loss: {:.6f}'.format(loss))

            global_step += 1
    

        if rank == 0:
            
            # Eval
            logger.info('============= Epoch: {} ============='.format(epoch))

            model.eval()

            loss_sum = 0
            with torch.no_grad():
                for batch_idx, (text_batch, prom_batch, code_batch) in enumerate(val_loader):
                    text_batch, prom_batch, code_batch =  map(lambda t: t.to(device), [text_batch, prom_batch, code_batch])

                    out, loss = model(text_batch, prom_batch, code_batch, infer=False)
                    
                    loss_sum += loss
            
            loss = loss_sum / len(val_loader)
            logger.info('Average Loss for {} Eval data: {:.6f}'.format(len(val_loader), loss))

                    # Infer
                    # out = model(text_batch, prom_batch, code_batch, infer=True)
                    # codes = out[:,None,:]
                    # wavs, _ = decode(codes)
                    # gts, _ = decode(code_list)
                    # for i in range(batch_size):
                    #     soundfile.write("{}{}/{}.recon.wav".format(log_dir, epoch, batch_size*batch_idx+i), wavs.cpu()[i, 0], sr)
                    #     soundfile.write("{}{}/{}.gt.wav".format(log_dir, epoch, batch_size*batch_idx+i), gts.cpu()[i, 0], sr)

            # Save
            utils.save_checkpoint(model, optimizer, lr, epoch, os.path.join(ckpt_dir, "G_{}.pth".format(epoch)))
            logger.info('======================================'.format(epoch))

if __name__ == "__main__":
    main()

