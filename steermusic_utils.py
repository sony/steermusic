import inspect
# from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
    VitsModel,
    VitsTokenizer,
)

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler,DDIMInverseScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import Attention

from diffusers.utils.torch_utils import randn_tensor
try:
    from diffusers.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline,AudioPipelineOutput
from diffusers import AudioLDM2Pipeline
from functools import partial
# if is_librosa_available():
import librosa
from diffusers.utils import logging,replace_example_docstring
import torchaudio
logger = logging.get_logger(__name__)  


class SpecifyGradient(torch.autograd.Function):
    """
    This code defines a custom gradient function using PyTorch's `torch.autograd.Function` class. It is particularly helpful when you want to manipulate gradients manually in a deep learning model that relies on automatic differentiation. The class is called `SpecifyGradient`, and contains two essential methods: `forward` and `backward`.

1. The `@staticmethod` decorator indicates that these are static methods and can be called on the class itself, without instantiating an object from the class.

2. The `forward` method takes two input arguments: `ctx` and `input_tensor`. `ctx` is a context object used to store information needed for backward computation. `input_tensor` is the input tensor to this layer in the neural network. The purpose of this method is to compute the forward pass and store any required information for the backward pass.

3. The `@custom_fwd` decorator is a user-defined decorator (not provided here) which presumably wraps or modifies the forward method in some way, most likely to add functionality like logging, error checking or other custom behavior.

4. Inside the `forward` method, the ground truth gradient `gt_grad` is saved using `ctx.save_for_backward()`. This stored information will be used later in the backward function. The forward function then returns a tensor of ones with the same device and data type as the input tensor. This tensor will be used in the backward pass as a scaling factor to adjust the gradients.

5. The `backward` method takes two input arguments: `ctx` and `grad_scale`. `ctx` is the same context object used in the forward pass. `grad_scale` is the gradient scaling factor used to adjust the gradients. The purpose of this method is to compute the gradient updates with respect to the input during backpropagation. 

6. The `@custom_bwd` decorator is another user-defined decorator (not provided here) which performs a similar role for the backward method as the `@custom_fwd` decorator does for the forward method.

7. Inside the `backward` method, the ground truth gradient `gt_grad` is retrieved from the saved tensors. It is then scaled by multiplying it with `grad_scale`. The method returns the scaled gradient `gt_grad` and `None`. The `None` value is returned because there are no gradients to compute for `gt_grad` with respect to the input tensor – it is assumed to be an external property that doesn't require gradient computation.

This custom gradient function can be used in situations where you need to have fine-grained control over the gradients in a neural network. For example, if you want to perform gradient clipping or apply noise to the gradients, you would use this `SpecifyGradient` function in place of a standard PyTorch layer.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor


class MyCrossAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # save text-conditioned attention map only
        # get attention map of ref
        if hidden_states.shape[0] == 4: 
            attn.hs = hidden_states[2:3]
        # get attention map of trg
        else:
            attn.hs = hidden_states[1:2]

        return hidden_states

class PConLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size


    def get_attn_pcon_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        if c == 256:
            res = int(res2/4)
            ref_noise_reshape = ref_noise.reshape(bs, res, 4, c).permute(0, 3, 1, 2)  # [B,C,T,F]
            trg_noise_reshape = trg_noise.reshape(bs, res, 4, c).permute(0, 3, 1, 2)
        elif c == 384:
            res = int(res2/2)
            ref_noise_reshape = ref_noise.reshape(bs, res, 2, c).permute(0, 3, 1, 2)  # [B,C,T,F] [1,384,64,2]
            trg_noise_reshape = trg_noise.reshape(bs, res, 2, c).permute(0, 3, 1, 2)
        elif c == 640:
            # In this case, frequency dim =1, we shuffle according to temporal dim
            res = int(res2/1)
            ref_noise_reshape = ref_noise.reshape(bs, res, 1, c).permute(0, 3, 1, 2) # [1,640,32,1] 
            trg_noise_reshape = trg_noise.reshape(bs, res, 1, c).permute(0, 3, 1, 2)
        else:
            print("[ERROR] Incorrect dim!")

        ref_noise_pooled = ref_noise_reshape
        trg_noise_pooled = trg_noise_reshape

        # Normalize feature dim
        ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1) # [1, 1280, 16, 16]
        trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

        ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1) # [1,T, F,C]

        patch_ids = np.random.permutation(ref_noise_pooled.shape[1])  # Random shuffle 256 channel index
        patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))] 
        patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

        ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1) # remove batch dim [B,T,F,C] -> [T,F,C]


        trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1) # [1,T, F,C]
        trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) # remove batch dim [B,T,F,C] -> [T,F,C]
        loss += self.PatchNCELoss(ref_sample, trg_sample).mean()  
        return loss
    
    def get_attn_cut_loss_org(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        
        ref_noise_reshape = ref_noise.permute(0, 2, 1)
        trg_noise_reshape = trg_noise.permute(0, 2, 1)

        # Down sample the attention maps
        for ps in self.patch_size:

            ref_noise_pooled = ref_noise_reshape
            trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1) 
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 1)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))] 
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 1)
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1)
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean()  
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size # 1
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1] # F
        ref_noise = ref_noise.detach()
    
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) # [T,1]
        ref_noise = ref_noise.unsqueeze(0) # [1,T,F,C]
        trg_noise = trg_noise.unsqueeze(0) # [1,T,F,C]
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise.view(batch_size,npatches,-1), trg_noise.view(batch_size,npatches,-1).transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T
        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))
        return loss


class AudioLDM2_pipe(nn.Module):
    def __init__(self, device, fp16, vram_O, hf_key=None, t_range=[0.05, 0.95]): 
        """
        The `__init__` method initializes the class and loads a Stable Diffusion model using the specified version number, 
        and also sets the precision of the model to either float16 or float32 depending on the `fp16` parameter. 
        It sets the device the model will run on based on the `device` parameter. 
        If a `hf_key` parameter is provided, it will use the defined checkpoint, otherwise it will use 'cvssp/audioldm2'.
        """
        super().__init__()

        self.device = device

        print(f'[INFO] loading AudioLDM2 diffusion...')

        if fp16==True:
            self.precision_t = torch.float16
        else:
            self.precision_t = torch.float32

        if hf_key is not None:
            print(f"[INFO] using personalized model ckpt: {hf_key}")
            self.repo_id = hf_key
        else:
            print("[INFO] using pre defined cvssp/audioldm2")
            self.repo_id = "cvssp/audioldm2"
        pipe = AudioLDM2Pipeline.from_pretrained(self.repo_id, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        # Prepare unet for extract attention map
        self.unet = self.prep_unet(self.unet)

        self.text_encoder_2 = pipe.text_encoder_2
        self.projection_model = pipe.projection_model
        self.language_model = pipe.language_model
        self.tokenizer_2 = pipe.tokenizer_2
        self.vocoder = pipe.vocoder
        self.feature_extractor = pipe.feature_extractor
        self.encode_prompt = pipe.encode_prompt
        self.mel_spectrogram_to_waveform = pipe.mel_spectrogram_to_waveform

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        # feature_extractor.requires_grad_(False)
        self.language_model.requires_grad_(False)
        self.projection_model.requires_grad_(False)
        self.vocoder.requires_grad_(False)


        self.scheduler = DDIMScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t)
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t)
        self.scheduler_sampling = DDIMScheduler.from_pretrained(self.repo_id, subfolder="scheduler", torch_dtype=self.precision_t)
    

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) 
        # self.alphas = self.cosine_noise_schedule(self.num_train_timesteps).to(self.device)
        print(f'[INFO] loaded AudioLDM2 diffusion!')

    def cosine_noise_schedule(self, num_train_timesteps, s=0.008):
        """Generates a cosine noise schedule for diffusion models."""
        t = torch.linspace(0, num_train_timesteps, num_train_timesteps)  # Time steps
        f_t = torch.cos(((t / num_train_timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]  # Normalize to start at 1
        return alphas_cumprod

    def prep_unet(self, unet):
        for name, params in unet.named_parameters():
            if 'attn1' in name: # self-attention
                params.requires_grad = True
            else:
                params.requires_grad = False

        # replace the fwd function
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention":
                module.set_processor(MyCrossAttnProcessor())
        return unet

    @torch.no_grad()
    def get_text_embeds(self, prompt,use_guidance=True,negative_prompt = ""):
        """
            The `get_text_embeds` method takes a text prompt as input and returns
            the embeddings of that prompt using the `text_encoder`.
        """
        prompt_embds,attention_mask,generated_prompt_embds = self.encode_prompt(
                prompt = prompt,
                device = self.device,
                num_waveforms_per_prompt = 1,
                do_classifier_free_guidance= use_guidance,
                negative_prompt=negative_prompt)
        
        return prompt_embds,generated_prompt_embds,attention_mask

    def predict_noise(self, prompt_embds,generated_prompt_embds, attention_mask, mel_spec, 
                      guidance_scale=100, as_latent=True, t=None, noise=None, cfg=True,scheduler=None,disable_grad=True):
        if as_latent:
            latents = mel_spec
        else:
            latents = self.encode_audio(mel_spec)

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        if disable_grad ==True:
            with torch.no_grad():
                if noise is None:
                    # add noise
                    noise = torch.randn_like(latents)

                # x_t = \sqrt(\alpha_t)x_0 + \sqrt(1-\alpha_t) \eps, where \eps is the noise.
                # latent here is x_0
                if scheduler is None:
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                else:
                    latents_noisy = scheduler.add_noise(latents, noise, t)

                latent_model_input = torch.cat([latents_noisy] * 2)
                # Save input tensors for UNet
                noise_pred = self.unet(latent_model_input, t, 
                                    encoder_hidden_states=generated_prompt_embds,
                                    encoder_hidden_states_1=prompt_embds,
                                    encoder_attention_mask_1=attention_mask)[0]
        else:
            if noise is None:
                # add noise
                noise = torch.randn_like(latents)

            # x_t = \sqrt(\alpha_t)x_0 + \sqrt(1-\alpha_t) \eps, where \eps is the noise.
            # latent here is x_0
            if scheduler is None:
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            else:
                latents_noisy = scheduler.add_noise(latents, noise, t)
                

            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            noise_pred = self.unet(latent_model_input, t, 
                                encoder_hidden_states=generated_prompt_embds,
                                encoder_hidden_states_1=prompt_embds,
                                encoder_attention_mask_1=attention_mask)[0]
        if cfg==True:
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        return noise_pred, t, noise

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        """
        The `produce_latents` method takes text embeddings and a set of latents as inputs, 
        and produces the corresponding latents for the given text prompts using a generative model.
        """
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents


    def latents_to_audios(self, latents, return_mel=False):
        # if scale:
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spec = self.vae.decode(latents).sample
        audio = self.mel_spectrogram_to_waveform(mel_spec)
        if return_mel == True:
            return audio, mel_spec
        else:
            audio = self.mel_spectrogram_to_waveform(mel_spec)
            return audio


    def encode_audio(self,mels):

        posterior = self.vae.encode(mels).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

  