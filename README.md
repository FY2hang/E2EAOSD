# E2EAOSD: An End-to-End Efficient Framework for Small Object Detection in Aerial Imagery

## Installation 


conda create -n torch_2_3_0_py310 python=3.10 anaconda

# Check existing environments 
conda env list

# Activate the environment 
conda activate torch_2_3_0_py310

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

pip install -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# Check if Torch and GPU are successfully loaded 
python check_torch_gpu.py

## Dataset Preparation 

Please organize the **VisDrone2019** dataset as follows:

```text
VisDrone2019/
  ├── VisDrone2019-DET-train/
  │   ├── annotations/
  │   ├── images/
  │   └── classes.txt
  ├── VisDrone2019-DET-val/
  │   ├── annotations/
  │   ├── images/
  │   └── classes.txt
  └── VisDrone2019-DET-test-dev/
      ├── annotations/
      ├── images/
      └── classes.txt

# syntax: CUDA_VISIBLE_DEVICES=<gpu_id> python train.py -c <yml_path> --seed=0
# example:
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/rtdetr_visdrone.yml --seed=0

# syntax: python train.py -c <yml_path> --test-only -r <checkpoint_path>
# example:
python train.py -c configs/rtdetr_visdrone.yml --test-only -r output/best.pth
