# LDM_CT

First, Download the pretrained Variational AutoEncoder(VAE) and Latent-Diffusion Model(LDM) for CT checkpoint for this [link](https://drive.google.com/drive/folders/1Q-6VF3If4GM5AaE3iF-I8KqQ43MOdqNz?usp=sharing). 

Make sure that the path of checkpoint. 

checkpoint of LDM: ./checkpoint/model_ckpt.pth 

checkpoint of VAE: ./checkpoint/vae_ldm8/vae_ckpt.pth

After that, execute the sample_ldm.py to perform CT denoising with my proposed algorithm.
You can see the sample(improved version of noise) of the quarter-dose CT and the noise difference between sample and quarter-dose CT in sample/denoising and sample/noise directory.

If you want to execute the sampling with colab, First download the LDM_CT into the Google Drive and connect with V100 GPU(at least or more superior GPU(ex) A100)) and execute above sample.ipynb. Maybe, you can change the directory /content/drive/MyDrive/diffusion-posterior-sampling/ into /content/drive/MyDrive/LDM_CT/ to execute normally.
