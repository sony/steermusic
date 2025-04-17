# Implementation of SteerMusic: Enhanced Musical Consistency for Zero-shot Text-Guided and Personalized Music Editing

This is implementation codes of the [paper](https://doi.org/10.48550/arXiv.2504.10826): SteerMusic: Enhanced Musical Consistency for Zero-shot Text-Guided and Personalized Music Editing.

Our demonstration page is available in [Demo](https://steermusic.pages.dev/).

## Environment setting

This code was tested with Python3.8.10, Pytorch 2.2.0+cu121. SteerMusic relies on a pretrained [AudioLDM2](https://github.com/haoheliu/AudioLDM2).

You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

If you encounter issues such as
`
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
`
please manually change `cached_download` to `hf_hub_download` in `diffusers/utils/dynamic_modules_utils.py`.

## SteerMusic for Zero-shot Text-guided Music Editing

To perform a corse-grained text-to-music editing, run 

```
python SteerMusic_edit.py --audio_path '/path/to/source/music/' --prompt 'target prompt' --prompt_ref 'source prompt' --output_dir '/output/path/' --guidance_scale 30
```

Example 
```
python SteerMusic_edit.py --audio_path "./audios/bach_anh114.wav" --prompt "Energetic guitar cover with a groovy, reverberant melody." --prompt_ref "Energetic piano cover with a groovy, reverberant melody." --guidance_scale 30  --weight_aug 3
```

## SteerMusic+ for Personalized Music Editing

SteerMusic+ relies on a fine-tuned personalized diffusion model. To fine-tune a personalized diffusion model, please refer to [DreamSound](https://github.com/zelaki/DreamSound). In this SteerMusic+ implementation, we plug-in SteerMusic+ to a DreamSound fine-tuned on a **AudioLDM2 checkpoint**. To personalize your music editing, please follow the fine-tune instruction provided in [DreamSound](https://github.com/zelaki/DreamSound) and obtain a checkpoint captured the desired musical concept token.


To perform a fine-grained personalized music editing, please run

```
python SteerMusic_personalized.py --audio_path '/path/to/source/music/' --prompt_ref 'source prompt with [emphasized] edit area, e.g., a recording of [piano] music' --concept 'target concept' --personalized_ckpt '/path/to/personalized/diffusion/ckpt/' --guidance_scale 15
```

This is an example command. We provide an example of fine-tuned DreamSound ckpt on [bouzouki] concept which can be downloaded via the [link](https://zenodo.org/records/15226658). The reference audio examples for bouzouki are available inside the folder `audios`. Please unzip the downloaded ckpt file and put to the path `./DreamSound/outputs_bouzouki/`, then execute the codes as below:

```
python SteerMusic_personalized.py --audio_path "./audios/bach_anh114.wav" --prompt_ref "Energetic [piano] cover with a groovy, reverberant melody." --concept 'bouzouki' --personalized_ckpt './Dreamsound/outputs_bouzouki/pipeline_step_100' --guidance_scale 20
```

## Evaluation Metrics

- For CLAP and LPAPS score, please refer to [CLAP](./eval/clap_score.py) and [LPAPS](./eval/lpaps_score.py). These codes are adapted from [AudioEditingCode](https://github.com/HilaManor/AudioEditingCode).

- For FAD scores, please refer to [fadtk](https://github.com/microsoft/fadtk)

- For CDPAM score, please refer to [CDPAM](./eval/CDPAM.py), which is adapted from [CDPAM_repo](https://github.com/pranaymanocha/PerceptualAudio).

- For CQT-1 PCC score, please refer to [CQT-1](./eval/CQT1_PCC.py)

## Q&A

### 1.  The edited results are not noticeable. What should I do?

If the editing effects are not noticeable, consider increasing the guidance scale. A higher guidance scale can strengthen the influence of the editing instructions, leading to more pronounced changes in the output. See Section 5 in our paper and our supplementary materials for the discussion of the trade-off between editing effects and original music content preservation.

### 2.  The edited results of SteerMusic+ significantly distort the source melody. What should I do?

One possible reason is that the personalized diffusion model may be overfitted. To mitigate this, try adjusting the number of fine-tuning steps used during personalization. For more details, please refer to Figure 11 in our paper.

## Acknowledgement

We acknowledge the following works for sharing their implementation code:

[Delta_denoising_score](https://github.com/ethanhe42/dds/tree/main); [Constrative_denoising_score](https://github.com/HyelinNAM/ContrastiveDenoisingScore); [DreamSound](https://github.com/zelaki/DreamSound); [AudioEditingCodes](https://github.com/HilaManor/AudioEditingCode); [AudioLDM2](https://github.com/haoheliu/AudioLDM2); 