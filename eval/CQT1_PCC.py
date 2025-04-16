import torch
import torchaudio
from nnAudio.features import CQT2010
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio.transforms as T

cqt = CQT2010(
    sr=16000,
    fmin=65,
    fmax=2100,
    n_bins=128,
    bins_per_octave=24,
    norm=1,
    basis_norm=1,
    window='hann',
    pad_mode='constant',
    earlydownsample=True
)

def cqt_topk_extractor(audio_path, topk=4,sr = 16000):
    audio_org, sr_org = torchaudio.load(audio_path)
    # In case that MusicCap is 44.1kHz sampling rate
    if sr_org != sr:
        resampler = T.Resample(orig_freq=sr_org, new_freq=sr)
        audio = resampler(audio_org)
    else:
        audio = audio_org
    # extracting the top 4 values and their indices from CQT
    value, index = cqt(audio)[0].topk(topk, dim=0)
    return index

def melody_pearsonr(audio_path1, audio_path2):
    # since the top-2 indices are very similar to top-1
    # we only use top-1 for comparison
    index1 = cqt_topk_extractor(audio_path1)[0]
    index2 = cqt_topk_extractor(audio_path2)[0]

    # if the length of the indices are not the same, we cut the longer one
    if len(index1) > len(index2):
        index1 = index1[:len(index2)]
    elif len(index1) < len(index2):
        index2 = index2[:len(index1)]

    return pearsonr(index1, index2)

