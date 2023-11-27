from functools import partial
import os
import argparse
import yaml
import math

import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.logger import get_logger
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

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
}

def get_optimizer(config, parameters):
    if config['optim']['optimizer'] == 'Adam':
        return optim.Adam(parameters, lr=config['optim']['lr'], weight_decay=config['optim']['weight_decay'],
                          betas=(config['optim']['beta1'], 0.999), amsgrad=config['optim']['amsgrad'],
                          eps=config['optim']['eps'])
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default = './configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default = './configs/vanila_config.yaml')
    parser.add_argument('--optim_config', type=str, default = './configs/optimizer.yaml')
    parser.add_argument('--model_vae_config', type=str, default='./configs/autoencoder/vae_config.yaml')
    parser.add_argument('--model_vae_ckpt', type=str, default='./checkpoint/vae_ldm8/vae_ckpt.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoint')
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
    
    # Load configurations
    model_config = load_yaml(args.model_config) # configs/model_config.yaml
    diffusion_config = load_yaml(args.diffusion_config) # diffusion_config.yaml
    task_config = load_yaml(args.task_config) # denoising_config.yaml
    optim_config = load_yaml(args.optim_config) # optimizer.yaml
    model_vae_config = load_yaml(args.model_vae_config)
   

    embed_dim = model_vae_config['model']['params']['embed_dim']
    ddconfig = model_vae_config['model']['params']['ddconfig']
    lossconfig = model_vae_config['model']['params']['lossconfig']

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma']
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    start_epoch = 0

    # Load model
    model = create_model(**model_config)
    model = model.to(device)

    vae_model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim)
    vae_model = vae_model.to(device)
    states = torch.load(args.model_vae_ckpt)
    vae_model.load_state_dict(states[0])
    vae_model.eval()
    for param in vae_model.parameters():
      param.requires_grad = False 

    print("VAE_MODEL_PATH: ", args.model_vae_ckpt)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    num_diffusion_timesteps = int(diffusion_config['steps'])
    betas = get_named_beta_schedule(
      schedule_name=diffusion_config['noise_schedule'],
      num_diffusion_timesteps=num_diffusion_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    dataset = get_dataset(**data_config)
    train_loader = get_dataloader(dataset, batch_size=1, num_workers=2, train=True)

    optimizer = get_optimizer(optim_config, model.parameters())

    if args.resume_training:
      print("Resume training Epoch", args.start_epoch)
      start_epoch = args.start_epoch
      states = torch.load(os.path.join(args.save_dir, str(start_epoch) + "_ckpt.pth"))
      model.load_state_dict(states[0])
      states[1]["param_groups"][0]["eps"] = optim_config["optim"]["eps"]
      optimizer.load_state_dict(states[1])
      assert states[2] == start_epoch
      start_epoch = states[2] + 1


    # Training
    for epoch in range(start_epoch, args.num_epochs):
      with tqdm(train_loader) as pbar:
        for i, batch in enumerate(pbar):
          model.train()
          batch = batch.to(device)
          posterior = vae_model.encode(batch)
          data = (posterior.sample()).to(device)
          n = data.size(0)
          e = torch.randn_like(data)
          t = torch.randint(
              low=0, high=num_diffusion_timesteps, size=(n // 2 + 1,)
          ).to(device)
          t = torch.cat([t, num_diffusion_timesteps - t - 1], dim=0)[:n]

          loss = loss_registry['simple'](model, data, t, e, betas)
          pbar.set_description(f'Epoch{epoch}')
          pbar.set_postfix(loss=loss)
          optimizer.zero_grad()
          loss.backward()
          try:
              torch.nn.utils.clip_grad_norm_(
                  model.parameters(), optim_config['optim']['grad_clip']
              )
          except Exception:
              pass
          optimizer.step()

      if epoch % args.interval_to_store == 0:
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
        ]
        torch.save(states, os.path.join(args.save_dir, str(epoch) + "_ckpt.pth"))


if __name__ == '__main__':
    main()