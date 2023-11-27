from functools import partial
import os
import argparse
from types import new_class
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio

from ldm.models.autoencoder import AutoencoderKL

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F2

std, mean = 490, -581

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default = './configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default = './configs/denoising_ldm_config.yaml')
    parser.add_argument('--optim_config', type=str, default = './configs/optimizer.yaml')
    parser.add_argument('--model_vae_config', type=str, default='./configs/autoencoder/vae_config.yaml')
    parser.add_argument('--model_vae_ckpt', type=str, default='./checkpoint/vae_ldm8/vae_ckpt.pth')
    parser.add_argument('--model_ckpt', type=str, default='./checkpoint/model_ckpt.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoint')
    parser.add_argument('--interval_to_store', type=int, default=10)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    model_vae_config = load_yaml(args.model_vae_config)

    embed_dim = model_vae_config['model']['params']['embed_dim']
    ddconfig = model_vae_config['model']['params']['ddconfig']
    lossconfig = model_vae_config['model']['params']['lossconfig']
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model_states = torch.load(args.model_ckpt)
    model.load_state_dict(model_states[0])
    model.eval()

    vae_model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim)
    vae_model = vae_model.to(device)
    vae_states = torch.load(args.model_vae_ckpt)
    vae_model.load_state_dict(vae_states[0])
    vae_model.eval()

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
    sample_fn = partial(sampler.p_sample_loop_vae, model=model, vae_model=vae_model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    dataset = get_dataset(**data_config)
    test_loader = get_dataloader(dataset, batch_size=1, num_workers=2, train=False)
    psnr_total = 0
    ssim_total = 0
    num_ct = 0
    # Do Inference
    for index, data in enumerate(test_loader):
      logger.info(f"Inference for image {index}")
      x, ref_img = data
      ref_img = ref_img.to(device)
      x = x.to(device)

      # Sampling
      x_start = torch.randn((1,2,128,128), device=device).requires_grad_()
      sample = sample_fn(x_start=x_start, measurement=ref_img, record=False, save_root=out_path)
      with torch.no_grad():
        recon_sample = vae_model.decode(sample)
        sample = recon_sample * std + mean
        x = x * std + mean
        new_sample = torch.cat([x, sample], dim=0)
        imgs = make_grid(tensor=new_sample, normalize=True, nrow=2, value_range=(-1024,3071))
        if not isinstance(imgs, list): # 하나의 이미지일때
          imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False) # 총 사진의 개수만큼 plot
        for i, img in enumerate(imgs):
            img = img.detach() # 학습 그래프에서 제외
            img = F2.to_pil_image(img) # torch.tensor 에서 pil 이미지로 변환
            axs[0, i].imshow(np.asarray(img)) # numpy 배열로 변경후, 가로로 이미지를 나열
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig(os.getcwd()+"/sample/denoising/" + str(index)+'.jpg') 
        
        ref_img = ref_img * std + mean
        noise_sample = ref_img - sample
        noise = ref_img - x
        new_noise = torch.cat([noise, noise_sample], dim=0)
        imgs2 = make_grid(tensor=new_noise, normalize=True, nrow=2, value_range=(-1,1))
        if not isinstance(imgs2, list): # 하나의 이미지일때
          imgs2 = [imgs2]
        fig, axs = plt.subplots(ncols=len(imgs2), squeeze=False) # 총 사진의 개수만큼 plot
        for i, img in enumerate(imgs2):
            img = img.detach() # 학습 그래프에서 제외
            img = F2.to_pil_image(img) # torch.tensor 에서 pil 이미지로 변환
            axs[0, i].imshow(np.asarray(img)) # numpy 배열로 변경후, 가로로 이미지를 나열
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig(os.getcwd()+"/sample/noise/" + str(index)+'.jpg') 

        original = torch.cat([x, ref_img], dim=0)
        imgs3 = make_grid(tensor=original, normalize=True, nrow=2, value_range=(-1024,3071))
        if not isinstance(imgs3, list): # 하나의 이미지일때
          imgs3 = [imgs3]
        fig, axs = plt.subplots(ncols=len(imgs3), squeeze=False) # 총 사진의 개수만큼 plot
        for i, img in enumerate(imgs3):
            img = img.detach() # 학습 그래프에서 제외
            img = F2.to_pil_image(img) # torch.tensor 에서 pil 이미지로 변환
            axs[0, i].imshow(np.asarray(img)) # numpy 배열로 변경후, 가로로 이미지를 나열
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig(os.getcwd()+"/sample/original/" + str(index)+'.jpg')         

        num_ct += 1

        data_high_condition = x > 3071
        data_low_condition = x < -1024

        x[data_high_condition] = 3071
        x[data_low_condition] = -1024

        recon_high_condition = sample > 3071
        recon_low_condition = sample < -1024

        sample[recon_high_condition] = 3071
        sample[recon_low_condition] = -1024 

        x = x.squeeze()
        sample = sample.squeeze()

        x = x.detach().cpu().numpy()
        sample = sample.detach().cpu().numpy()

        psnr_val = peak_signal_noise_ratio(x, sample, data_range=4096)
        psnr_total += psnr_val
        print("AVG_PSNR: ", psnr_total/num_ct)


        ssim_val = ssim(x, sample, data_range=4096)
        ssim_total += ssim_val
        print("AVG_SSIM: ", ssim_total/num_ct)

        


if __name__ == '__main__':
    main()
