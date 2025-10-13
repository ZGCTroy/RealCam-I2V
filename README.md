# RealCam-I2V

<div align="center">
    <a href="https://arxiv.org/abs/2502.10059"><img src="https://img.shields.io/static/v1?label=arXiv&message=2502.10059&color=b21d1a"></a>
    <a href="https://zgctroy.github.io/RealCam-I2V"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a>
    <a href="https://github.com/ZGCTroy/RealCam-Vid"><img src="https://img.shields.io/static/v1?label=Dataset&message=RealCam-Vid&color=blue"></a>
</div>

[ICCV'25] Official repo of "RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control".

## 🌟 News

- 25/07/05: Release inference code and checkpoints of RealCam-I2V on CogVideoX 1.5 for exploration. The results we report in the [paper](https://arxiv.org/abs/2502.10059) are based on DynamiCrafter, for full reproduction and [evaluation](https://github.com/ZGCTroy/CamI2V/tree/main/evaluation), please refer to our previous repo [CamI2V](https://github.com/ZGCTroy/CamI2V).
- 25/06/26: RealCam-I2V is accepted by ICCV 2025! 🎉🎉
- 25/05/18: Release training code of RealCam-I2V on CogVideoX 1.5.
- 25/03/26: Release our dataset [RealCam-Vid](https://huggingface.co/datasets/MuteApo/RealCam-Vid) v1 for metric-scale camera-controlled video generation!
- 25/02/18: Initial commit of the project, we plan to release our DiT-based real-camera I2V models (e.g., CogVideoX) in this repo.

## ⚙️ Environment

### Quick Start

```shell
apt install libgl1-mesa-glx libgl1-mesa-dri xvfb # for ubuntu
yum install -y mesa-libGL mesa-dri-drivers Xvfb. # for centos
conda install ffmpeg=7 -c conda-forge
pip install -r requirements.txt
```

## 💫 Inference

### Download Pretrained Models

Download and put under `pretrained` folder the pretrained weights of [CogVideoX1.5-5B-I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V), [Metric3D](https://huggingface.co/JUGGHM/Metric3D) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

### Download Model Checkpoints

Download our weights of [RealCam-I2V](https://huggingface.co/MuteApo/RealCam-I2V) and put under `checkpoints` folder.
Please edit `demo/models.json` if you have a custom model path.

### Run Gradio Demo

```shell
python gradio_app.py
```

## 🚀 Training

### Prepare Dataset

Please access [RealCam-Vid](https://github.com/ZGCTroy/RealCam-Vid) and download our dataset for training RealCam-I2V-CogVideoX-1.5. Please unzip all contents in `data` folder.

### Launch

Edit example training script `accelerate_train.sh` if necessary and launch training by:

```shell
bash accelerate_train.sh
```

## 🤗 Related Repo

- Our dataset, the first open-sourced, combining diverse scene dynamics with metric-scale camera trajectories, is available at [RealCam-Vid](https://github.com/ZGCTroy/RealCam-Vid).
- Our previous work at [CamI2V](https://github.com/ZGCTroy/CamI2V).
- We have borrowed a lot of code from the original [CogVideoX](https://github.com/THUDM/CogVideo) repository.

## 🗒️ Citation

```
@article{li2025realcam,
    title={RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control}, 
    author={Li, Teng and Zheng, Guangcong and Jiang, Rui and Zhan, Shuigen and Wu, Tao and Lu, Yehao and Lin, Yining and Li, Xi},
    journal={arXiv preprint arXiv:2502.10059},
    year={2025},
}

@article{zheng2024cami2v,
    title={CamI2V: Camera-Controlled Image-to-Video Diffusion Model},
    author={Zheng, Guangcong and Li, Teng and Jiang, Rui and Lu, Yehao and Wu, Tao and Li, Xi},
    journal={arXiv preprint arXiv:2410.15957},
    year={2024}
}
```
