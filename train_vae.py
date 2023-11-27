import os
import numpy as np
import torch
import yaml
import argparse
import torch.optim as optim

from util.logger import get_logger
from data.dataloader import get_dataset, get_dataloader
from tqdm.auto import tqdm
from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F2

std, mean = 490, -581

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_optimizer(config, parameters):
    if config['optim']['optimizer'] == 'Adam':
        return optim.Adam(parameters, lr=config['optim']['lr'], weight_decay=config['optim']['weight_decay'],
                          betas=(config['optim']['beta1'], 0.999), amsgrad=config['optim']['amsgrad'])
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str, default='./configs/autoencoder/vae_config.yaml')
  parser.add_argument('--task_config', type=str, default = './configs/vanila_config.yaml')
  parser.add_argument('--optim_config', type=str, default = './configs/autoencoder/vae_optimizer.yaml')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--save_dir', type=str, default='./checkpoint/vae_ldm8')
  parser.add_argument('--num_epochs', type=int, default=500)
  parser.add_argument('--resume_training', action="store_true")
  parser.add_argument('--start_epoch', type=int, default=None)
  parser.add_argument('--interval_to_store', type=int, default=5)
  args = parser.parse_args()

  # logger
  logger = get_logger()
  
  # Device setting
  device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
  logger.info(f"Device set to {device_str}.")
  device = torch.device(device_str) 
  logger.info(f"Model config: {args.model_config}")
  logger.info(f"Optim config: {args.optim_config}")
  logger.info(f"Save dir: {args.save_dir}")

  model_config = load_yaml(args.model_config)
  task_config = load_yaml(args.task_config)
  optim_config = load_yaml(args.optim_config)

  embed_dim = model_config['model']['params']['embed_dim']
  ddconfig = model_config['model']['params']['ddconfig']
  lossconfig = model_config['model']['params']['lossconfig']

  data_config = task_config['data']
  dataset = get_dataset(**data_config)
  train_loader = get_dataloader(dataset, batch_size=1, num_workers=2, train=True)

  model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim)
  model = model.to(device)

  loss_fn = instantiate_from_config(lossconfig).to(device)

  ae_optimizer = get_optimizer(optim_config, model.parameters())
  disc_optimizer = get_optimizer(optim_config, loss_fn.discriminator.parameters())

  start_epoch = 0
  global_step = 0

  if args.resume_training:
    start_epoch = args.start_epoch
    states = torch.load(os.path.join(args.save_dir, str(start_epoch) + "_ckpt.pth"))
    model.load_state_dict(states[0])
    ae_optimizer.load_state_dict(states[1])
    disc_optimizer.load_state_dict(states[2])
    loss_fn.discriminator.load_state_dict(states[3])
    assert states[4] == start_epoch
    start_epoch = states[4] + 1
    global_step = states[5]

  print("lr: ", ae_optimizer.param_groups[-1]['lr'])
  print("global step: ", global_step)
  for epoch in range(start_epoch, args.num_epochs):
    with tqdm(train_loader) as pbar:
      for i,data in enumerate(pbar):
        model.train()
        data = data.to(device)
        reconstructions, posterior = model(data)

        aeloss, log_dict_ae = loss_fn(data, reconstructions, posterior, 0, global_step,
                                        last_layer=model.get_last_layer(), split="train")
                                        
        ae_optimizer.zero_grad()
        aeloss.backward()
        ae_optimizer.step()

        discloss, log_dict_disc = loss_fn(data, reconstructions, posterior, 1, global_step,
                                           last_layer=model.get_last_layer(), split="train")

        disc_optimizer.zero_grad()
        discloss.backward()
        disc_optimizer.step()

        global_step += 1
        pbar.set_description(f'Epoch{epoch}')
        pbar.set_postfix(aeloss=aeloss, discloss=discloss)

    if epoch % args.interval_to_store == 0:
      states = [
          model.state_dict(),
          ae_optimizer.state_dict(),
          disc_optimizer.state_dict(),
          loss_fn.discriminator.state_dict(),
          epoch,
          global_step,
      ]
      torch.save(states, os.path.join(args.save_dir, str(epoch) + "_ckpt.pth"))
        
if __name__ == '__main__':
    main()