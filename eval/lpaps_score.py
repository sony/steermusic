import torch
import numpy as np
from meta_clap_consistency import CLAPTextConsistencyMetric
import torchaudio
from typing import Optional
import os
from datasets import load_dataset
from tqdm import tqdm
import pdb

from lpaps import LPAPS

clap_checkpoint_path =  'clap/pretrained'
clap_ckpt_name = 'music_audioset_epoch_15_esc_90.14.pt'

device = 'cuda:0'


lpaps_model = LPAPS(net='clap', device=device,
                        net_kwargs={'model_arch': 'HTSAT-base' if 'fusion' not in clap_ckpt_name
                                    else 'HTSAT-tiny',
                                    'chkpt': clap_ckpt_name,
                                    'enable_fusion': 'fusion' in clap_ckpt_name},
                        checkpoint_path=clap_checkpoint_path).to(device)


def compute_lpaps_with_windows(aud1: torch.Tensor, aud1_sr: int, aud2: torch.Tensor, aud2_sr: int, model: LPAPS,
                               windows_size1: Optional[int] = None, windows_size2: Optional[int] = None,
                               overlap: float = 0.1, method: str = 'mean', device: str = 'cuda:0') -> float:
    """Calculate the LPAPS score for the given audio files, windowed. If windows_size1 or windows_size2 is None, it will default to 10 seconds.

    :param torch.Tensor aud1: The first audio file to compute LPAPS for
    :param int aud1_sr: The sample rate of the first audio file
    :param torch.Tensor aud2: The second audio file to compute LPAPS for
    :param int aud2_sr: The sample rate of the second audio file
    :param LPAPS model: The LPAPS model to use
    :param Optional[int] windows_size1: Window size in seconds for the first audio file. Defaults to 10 seconds (None)
    :param Optional[int] windows_size2: Window size in seconds for the second audio file. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined LPAPS score
    """

    if windows_size1 is None:
        windows_size1 = int(aud1_sr * 10)
    if windows_size2 is None:
        windows_size2 = int(aud2_sr * 10)

    scores = []
    for i, j in zip(range(0, aud1.shape[-1], int(windows_size1 * (1 - overlap))),
                    range(0, aud2.shape[-1], int(windows_size2 * (1 - overlap)))):
        window1 = aud1[:, i:i + windows_size1]
        window2 = aud2[:, j:j + windows_size2]
        scores.append(model(window1.unsqueeze(0).to(device), window2.unsqueeze(0).to(device),
                      torch.tensor([aud1_sr], device=device),
                      torch.tensor([aud2_sr], device=device)).item())

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


def calc_lpaps_win(lpaps_model: LPAPS, aud1: torch.Tensor, aud2: torch.Tensor, sr1: int, sr2: int,
                   win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the LPAPS score between two audio files, with optional windowing

    :param LPAPS lpaps_model: An initialized LPAPS model to use
    :param torch.Tensor aud1: First audio file
    :param torch.Tensor aud2: Second audio file
    :param int sr1: Sample rate of the first audio file
    :param int sr2: Sample rate of the second audio file
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The LPAPS score
    """
    if win_length is None:
        return lpaps_model(aud1.unsqueeze(0).to(device),
                           aud2.unsqueeze(0).to(device),
                           torch.tensor([sr1], device=device),
                           torch.tensor([sr2], device=device)).item()
    else:
        return compute_lpaps_with_windows(aud1, sr1, aud2, sr2, lpaps_model,
                                          windows_size1=win_length * sr1,
                                          windows_size2=win_length * sr2,
                                          overlap=overlap, method=method, device=device)


def compute_lpaps_score(source_audio_path,target_audio_path,method = 'mean'):
    aud1, sr1 = torchaudio.load(source_audio_path)
    aud2, sr2 = torchaudio.load(target_audio_path)
    score = calc_lpaps_win(lpaps_model = lpaps_model, aud1 = aud1, aud2 = aud2, sr1= sr1, sr2 = sr2,
                  win_length = 10 if 'fusion' not in clap_ckpt_name else None,
                  method = method, overlap=0.1, device=device)
    return score

# score = compute_lpaps_score(source_audio_path=source_audio_name,target_audio_path = target_audio_name, method = "mean")