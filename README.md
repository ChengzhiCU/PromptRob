
# Zero shot Adv rob exp:

dataset list:
TODO:
STL10,
Caltech101,
Stanford Cars,
SUN397,
Caltech 101,
ImageNet,


CUDA_VISIBLE_DEVICES=5,6 python main_clip_addprompttoken.py --dataset cifar100 --root /local/vondrick/chengzhi/datasets --add_prompt_size 0

# compare inside prompt vs. outside token prompt.

# get multiGPU work

https://github.com/openai/CLIP/issues/111

### Find the visual prompt is fiting to the one hot embed, not the text. Which will cause overfit to one hot and lost ability to be zero shot.

# Visual Prompting
This is the official implementation of the paper [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274). 

![](./figures/clip_vs_vision.png)


## Installation
Clone this repo:
```bash
git clone https://github.com/hjbahng/visual_prompting.git
cd visual_prompting
```

This code requires python 3+. Install dependencies by:
```bash
pip install -r requirements.txt
```

Prepare the pre-trained models:
```bash
bash models/download_models.sh
```

## Train/Test for CLIP
* Train a visual prompt:
```bash
python main_clip.py --dataset cifar100 --root [path_to_cifar100] 
```

* Test the visual prompt:
```bash
python main_clip.py --evaluate --resume /path/to/checkpoints/model_best.pth.tar --dataset cifar100 --root [path_to_cifar100]
```

## Train/Test for Vision Models
* Train a visual prompt:
```bash
python main_vision.py --model bit_m_rn50 --dataset cifar100 --root [path_to_cifar100]
```

* Test the visual prompt:
```bash
python main_vision.py --evaluate --resume /path/to/checkpoints/model_best.pth.tar --model bit_m_rn50 --dataset cifar100 --root [path_to_cifar100]
``` 
* There are three model choices: `rn50`, `instagram_resnext101_32x8d`, and `bit_m_rn50`.
* Note that we use `--batch_size 32` for `instagram_resnext101_32x8d` and `--batch_size 128` for other models.

## Citation
If you use this code for your research, please cite our paper.
```
@article{bahng2022visual,
         title={Exploring Visual Prompts for Adapting Large-Scale Models}, 
         author={Hyojin Bahng and Ali Jahanian and Swami Sankaranarayanan and Phillip Isola},
         journal={arXiv preprint arXiv:2203.17274},
         year={2022}
}
```
