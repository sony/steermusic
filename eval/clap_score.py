import torch
import numpy as np
from meta_clap_consistency import CLAPTextConsistencyMetric
import torchaudio
from typing import Optional
import os
from datasets import load_dataset
from tqdm import tqdm
import pdb
clap_checkpoint_path =  'clap/pretrained'
clap_ckpt_name = 'music_audioset_epoch_15_esc_90.14.pt'

device = 'cuda:0'

clap_model = CLAPTextConsistencyMetric(model_path=os.path.join(clap_checkpoint_path, clap_ckpt_name),
                                        model_arch='HTSAT-base' if 'fusion' not in clap_ckpt_name else 'HTSAT-tiny',
                                        enable_fusion='fusion' in clap_ckpt_name
                                        ).to(device)
clap_model.eval()


def compute_clap_with_windows(aud: torch.Tensor, aud_sr: int, prompt: str, model: CLAPTextConsistencyMetric,
                              windows_size: Optional[int] = None, overlap: float = 0.1,
                              method: str = 'mean', device: torch.device = 'cuda:0') -> float:
    """Calculate the CLAP score for the given audio file and prompt, windowed. If windows_size is None, it will default to 10 seconds. 

    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int aud_sr: The sample rate of the audio file
    :param str prompt: The prompt to compute CLAP relative to
    :param CLAPTextConsistencyMetric model: The CLAP model to use
    :param Optional[int] windows_size: Window size in seconds. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined CLAP score
    """
    if windows_size is None:
        windows_size = int(aud_sr * 10)
    scores = []
    for i in range(0, aud.shape[-1], int(windows_size * (1 - overlap))):
        window = aud[:, i:i + windows_size]
        model.update(window.unsqueeze(0).to(device), [prompt], torch.tensor([aud_sr], device=device))
        scores.append(model.compute())
        model.reset()
    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)



def calc_clap_win(clap_model: CLAPTextConsistencyMetric, aud: torch.Tensor, sr: int, target_prompt: str,
                  win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the CLAP score between an audio file and a prompt, with optional windowing

    :param CLAPTextConsistencyMetric clap_model: An initialized CLAP model to use
    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int sr: The sample rate of the audio file
    :param str target_prompt: The prompt to compute CLAP relative to
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The CLAP score
    """
    if win_length is None:
        clap_model.update(aud.unsqueeze(0).to(device), [target_prompt], torch.tensor([sr], device=device))
        tmp = clap_model.compute()
        clap_model.reset()
        return tmp
    else:
        return compute_clap_with_windows(
            aud, sr, target_prompt, clap_model, device=device,
            windows_size=win_length * sr, overlap=overlap, method=method)

def compute_clap_score(audio_path,target_prompt,method = 'mean'):
    aud, aud_sr = torchaudio.load(audio_path)

    score = calc_clap_win(clap_model = clap_model, aud = aud, sr= aud_sr, target_prompt = target_prompt,
                  win_length = None,
                  method = method, overlap=0.1, device=device)
    return score


# score = compute_clap_score(audio_path=audio_name,target_prompt = prompt, method = "mean")