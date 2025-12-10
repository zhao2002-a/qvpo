# [NeurIPS 2024] Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization

Code release for **Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization (NeurIPS 2024)**.

[[paper]](https://arxiv.org/abs/2405.16173) [[project page]](https://dingsht.tech/qvpo-webpage/)

![](./asset/qvpo.png)

## Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `qvpo` can be created and activated with:
```
conda create qvpo
conda activate qvpo
```
To get started, install the additionally required python packages into you environment.
```
pip install -r requirements.txt
```

## Running
Running experiments based our code could be quite easy, so below we use `HalfCheetah-v3` task as an example. 

```
python main.py --env_name HalfCheetah-v3 --weighted --aug
```

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{
ding2024diffusionbased,
title={Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization},
author={Shutong Ding and Ke Hu and Zhenhao Zhang and Kan Ren and Weinan Zhang and Jingyi Yu, Jingya Wang and Ye Shi},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://arxiv.org/abs/2405.16173}
}
```

## Acknowledgement

The code of QVPO is based on the implementation of [DIPO](https://github.com/BellmanTimeHut/DIPO).


## Additional Notes on Code Replication
1. Better set python version to 3.10
2. Failed to build 'gym'? Solution: 
```
python3 -m pip install "pip<24" "setuptools<60" "wheel<0.38"
python3 -m pip install gym==0.21.0 --no-build-isolation
python3 -m pip install -r requirements.txt
```
3. Error related to `pygame 2.1.0`? Solution:
```
sudo apt update
sudo apt install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
                 libportmidi-dev libfreetype6-dev libjpeg-dev libpng-dev libsmpeg-dev \
                 libavformat-dev libswscale-dev libgl1-mesa-dev libglu1-mesa-dev
```
4. Failed to build `box2d-py`? Solution:
```
sudo apt update && sudo apt install swig
```
5. Download Mujoco package and add it to the `$PATH` in `~/.bashrc`
6. Compiling mujoco-py requires `sudo apt install libglew-dev patchelf`