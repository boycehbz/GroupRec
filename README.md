# GroupRec
# \[ICCV2023\] Reconstructing Groups of People with Hypergraph Relational Reasoning (GroupRec)

The official code for ICCV 2023 paper "Reconstructing Groups of People with Hypergraph Relational Reasoning"<br>
[Buzhen Huang](http://www.buzhenhuang.com/), [Jingyi Ju](https://me-ditto.github.io/), [Zhihao Li](https://scholar.google.com/citations?user=4cuefJ0AAAAJ&hl=zh-CN&oi=ao), [Yangang Wang](https://www.yangangwang.com/)<br>
\[[Project](https://www.yangangwang.com/papers/iccv2023-grouprec/HUANG-GROUPREC-2023-07.html)\] \[[Paper](https://arxiv.org/abs/2308.15844)\] \[[Video](https://www.bilibili.com/video/BV1Up4y1j7PY/?spm_id_from=888.80997.embed_other.whitelist&t=3.205652&bvid=BV1Up4y1j7PY&vd_source=8737daf9cd6bef1969ee7438f3bd08c7)\]

![figure](/assets/pipeline.jpg)

## Installation 
Create conda environment and install dependencies.
```
conda create -n grouprec python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 # install pytorch
pip install -r requirements.txt
```
## Getting Started
**Step1:**<br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in ```data/SMPL_NEUTRAL.pkl```.<br>


**Step2:**<br>
Download trained models from [Baidu Netdisk](https://pan.baidu.com/s/14BD-i_wUBV_wEh3l1yo0IQ?pwd=tucv) and put them in ```data```.<br>

**Step3:**<br>
Run demo for 3D pose estimation.
```bash
python demo.py --config cfg_files/demo.yaml
```

Run demo for SMPL estimation.
```bash
python demo.py --config cfg_files/demo_smpl.yaml
```

## Train
You can download the training data from [Baidu Netdisk](https://pan.baidu.com/s/1stGS_zFsAFgnr0QmKw6wGA?pwd=uthy) and place it in the ```data``` directory. Due to licensing restrictions, we cannot provide the Human3.6M dataset. However, you may process it into the same format as our provided data.
```bash
python main.py --config cfg_files/config.yaml
```




## Pseudo Dataset
We provide pseudo annotations for Panda dataset (Detection and MOT). You may also need to download image files from their official websites.

\[[Annotations](https://pan.baidu.com/s/1b8_aXe2RCJQbNLA1zQ_r8w?pwd=vy3j)\]
\[[Detection Image](https://www.gigavision.cn/track/track?nav=Detection&type=nav&t=1696068967354)\]
\[[MOT Image](https://www.gigavision.cn/track/track?nav=Tracking&type=nav&t=1696069113370)\]

## TODOS:

- [x] Demo code for pose estimation
- [x] Demo code for SMPL estimation
- [x] Training code release


## Citation
If you find this code useful for your research, please consider citing the paper.
```
@inproceedings{grouprec,
title={Reconstructing Groups of People with Hypergraph Relational Reasoning},
author={Huang, Buzhen and Ju, Jingyi and Li, Zhihao and Wang, Yangang},
booktitle={ICCV},
year={2023},
}
```

## Acknowledgments
Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.<br>
[CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)<br>
[ByteTrack](https://github.com/ifzhang/ByteTrack)<br>
[LoCO](https://github.com/fabbrimatteo/LoCO)<br>
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br>
