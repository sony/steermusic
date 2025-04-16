import essentia
import essentia.standard
import librosa
import pdb
from scipy.spatial.distance import cosine
import numpy as np
import os
from datasets import load_dataset
from tqdm import tqdm
import cdpam
import torch

loss_fn = cdpam.CDPAM(dev='cuda:0')

def calculate_cdpam(audio1,audio2,target_sr=16000):
    y1 = cdpam.load_audio(audio1)
    y2 = cdpam.load_audio(audio2)
    with torch.no_grad():
        dist = loss_fn.forward(y1,y2)
    return dist.item()
