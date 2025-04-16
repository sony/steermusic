# %%
import torch
from tqdm import tqdm
import steermusic_utils
import numpy as np
import os
import yaml
from scipy.io.wavfile import write
import torch.nn.functional as F
from preprocessor import Preprocessor
import argparse

device = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', type=str)
parser.add_argument('--prompt', type=str)
parser.add_argument('--prompt_ref', default='', type=str)
parser.add_argument('--output_dir', default='./SteerMusic_output/', type=str)
parser.add_argument('--validation_step', default=400, type=int)
parser.add_argument('--guidance_scale', default=30, type=int)
parser.add_argument('--weight_aug', default=2, type=int)
opt = parser.parse_args()

audio_name = opt.audio_path
prompt_ref = opt.prompt_ref
prompt = opt.prompt
weight_aug = opt.weight_aug
guidance_scale = opt.guidance_scale
validation_step = opt.validation_step
# Result store path
output_dir = opt.output_dir
os.makedirs(output_dir, exist_ok=True)


# Read config
config_yaml = f"config/autoencoder/16k_64.yaml"
exp_name = os.path.basename(config_yaml.split(".")[0])
exp_group_name = os.path.basename(os.path.dirname(config_yaml))

config_yaml = os.path.join(config_yaml)
config_yaml = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

# Obtain preprocessor
preprocessor = Preprocessor(config_yaml)

# Load guidance model
guidance_model = steermusic_utils.AudioLDM2_pipe(device, fp16=False, vram_O=False, t_range=[0.02, 0.98]) # You can adjust t_range
guidance_model.eval()


def SteerMusic_editing(audio_name,prompt_ref,prompt,weight_aug,validation_step,guidance_model=guidance_model, preprocessor=preprocessor,guidance_scale = 15, device = "cuda:7"):
    log_mel_spec, _, _, _  = preprocessor.read_audio_file(filename=audio_name)
    log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(device)

    for p in guidance_model.parameters():
        p.requires_grad = False

    log_mel_spec.requires_grad = False

    # Obtain latent size at first 
    lt = guidance_model.encode_audio(log_mel_spec)
    as_latent = True
    latent_size = 16 # need to adjust!
    latent = torch.randn(1, 8, lt.shape[2], latent_size, requires_grad=True, device=device)

    with torch.no_grad():
        print(f'[INFO] Encode text prompt...')
        text_z_ref, text_z_ref_gen,z_attention_mask_ref = guidance_model.get_text_embeds(prompt_ref)
        text_z,text_z_gen,z_attention_mask = guidance_model.get_text_embeds(prompt)
        print(f'[INFO] Encoding audio into latent...')
        latent_ref = guidance_model.encode_audio(log_mel_spec) 
        latent[:] = latent_ref


    optim = torch.optim.SGD([latent], lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)


    print(f'[INFO] Delta denoising score...')
    for i in tqdm(range(validation_step+1)):
        optim.zero_grad()
        x = latent


        noise_pred, t, noise = guidance_model.predict_noise(prompt_embds=text_z,
                                                            generated_prompt_embds=text_z_gen,
                                                            attention_mask = z_attention_mask,
                                                            mel_spec = x,
                                                            guidance_scale=guidance_scale, 
                                                            as_latent=as_latent)

        with torch.no_grad():
            noise_pred_ref, _, _ = guidance_model.predict_noise(prompt_embds=text_z_ref,
                                                        generated_prompt_embds=text_z_ref_gen,
                                                        attention_mask = z_attention_mask_ref,
                                                        mel_spec = latent_ref,
                                                        guidance_scale=guidance_scale, 
                                                        as_latent=as_latent,
                                                        t=t,
                                                        noise=noise)
        # validation
        if i % validation_step == 0:
            with torch.no_grad():
                # print("cos sim at step", i, " :", F.cosine_similarity(noise_pred - noise, noise_pred_ref - noise).mean().item())
                if as_latent:
                    audio = guidance_model.latents_to_audios(x)
                    basename = os.path.basename(audio_name)
                    write(f"./{output_dir}/{basename}",16000, audio[0].detach().cpu().numpy())
        
        w =  weight_aug*(1 - guidance_model.alphas[t])
        grad = w * (noise_pred - noise_pred_ref)
        grad = torch.nan_to_num(grad)

        loss = steermusic_utils.SpecifyGradient.apply(x, grad)
        loss.backward()
        optim.step()
        scheduler.step()



SteerMusic_editing(audio_name,prompt_ref,prompt,weight_aug,validation_step,guidance_model= guidance_model,
                   preprocessor= preprocessor,guidance_scale= guidance_scale,device=device)
