<div align="center">
<h1> PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection</h1>

Xiaoyu Zhou<sup>1</sup>, Jianwei Fei<sup>2</sup>, Peipeng Yu<sup>1</sup>, Jingchang Xie<sup>3</sup>, Chong Cheng<sup>1</sup>, Zhihua Xia<sup>1✉</sup>

<sup>1</sup>College of Cyber Security, Jinan University, Guangzhou, China<br>
<sup>2</sup>Department of Information Engineering, University of Florence, Florence, Italy<br>
<sup>3</sup>School of Integrated Circuits, Guangdong University of Technology, Guangzhou, China<br>
</div>


## 🤖 方法 / 架构

<div style="text-align:center; margin:20px 0;">
    <img src="assets/framework_v2.png" style="max-width:80%; height:auto;" alt="PGC Framework" />
    <br>
    <em>Overview of the Peak-Guided Calibration (PGC) Framework.</em>
</div>


## 📣 News

- `2026/05` : 🎉 我们的论文已被 ICML 2026 接收！

---

## 🛠️ 1. 环境配置
```bash
conda create -n pgc python=3.10 -y
conda activate pgc
pip install -r requirements.txt
```


## 📂 2. 数据集与权重

请在开始训练或测试之前，按以下分类下载所需的数据集、代码和模型权重。

### 2.1 CommGen15 和模型权重
以下为本论文提出的预训练模型与评估数据集，已统一发布至 ModelScope 和 Hugging Face：
- **CommGen15 数据集 (Dataset)**: [ModelScope-CommGen15](https://modelscope.cn/datasets/xiaoyuzhou68/CommGen15) 和 [HuggingFace-CommGen15](https://huggingface.co/datasets/xiaoyuzhou68/CommGen15)
- **PGC 预训练模型合集 (3 个 Checkpoints)**: [ModelScope-PGC_ckpt](https://modelscope.cn/models/xiaoyuzhou68/PGC_ckpt) 和 [HuggingFace-PGC_ckpt](https://huggingface.co/xiaoyuzhou68/PGC_ckpt)

### 2.2 基础模型与评估基准
除了本论文的数据外，还需要下载 DINOv2 基础模型以及开源的基准数据集进行全面的对比和实验：
- **DINOv2-Large 预训练权重**: [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large)
- **AIGI 基准测试**: [HorizonTEL/AIGIBench (GitHub)](https://github.com/HorizonTEL/AIGIBench)
- **GenImage 基准测试**: [GenImage-Dataset/GenImage (GitHub)](https://github.com/GenImage-Dataset/GenImage)
- **UniversalFakeDetect 基准测试**: [WisconsinAIVision/UniversalFakeDetect (GitHub)](https://github.com/WisconsinAIVision/UniversalFakeDetect)

### 2.3 目录结构配置 (Directory Structure)

下载完毕后，请按以下结构组织相应的模型和数据集：

**模型权重目录：**
```text
<PROJECT_ROOT>/
├── pretrained_model/
│   ├── PGC_train_progan_ckpt.pth
│   ├── PGC_train_progan_sdv1_4_ckpt.pth
│   └── PGC_train_sdv1_4_ckpt.pth
└── pretrained_dino/
    └── dinov2-large/
        ├── config.json
        ├── model.safetensors
        └── ...
```

**测试集目录结构：**
测试数据集需要保持以下的目录格式，程序将自动读取真实图片（`0_real`，标签为 $0$）和伪造图片（`1_fake`，标签为 $1$）：

```text
<DATA_ROOT>/GenImage/test/
├── ADM/
│   ├── 0_real/
│   └── 1_fake/
├── BigGAN/
│   ├── 0_real/
│   └── 1_fake/
├── glide/
│   ├── 0_real/
│   └── 1_fake/
├── Midjourney/
│   ├── 0_real/
│   └── 1_fake/
├── stable_diffusion_v_1_4/
│   ├── 0_real/
│   └── 1_fake/
├── stable_diffusion_v_1_5/
│   ├── 0_real/
│   └── 1_fake/
├── VQDM/
│   ├── 0_real/
│   └── 1_fake/
└── wukong/
    ├── 0_real/
    └── 1_fake/
```

## 🚀 3. 训练指令 (Training Commands)

我们提供以下三种训练设置，您可以根据需求选择相应的训练脚本：

### 3.1 使用 SDv1.4 训练集进行训练

```bash
python train.py \
  --name pgc_train_sdv1_4 \
  --checkpoints_dir checkpoints \
  --devices 0 \
  --batch_size 32 \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --lora_dropout 0.1 \
  --real_image_dir <DATA_ROOT>/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature \
  --fake_image_dir <DATA_ROOT>/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai \
  --test_root <DATA_ROOT>/GenImage/test \
  --cropSize 224 \
  --niter 100 \
  --lr 5e-5 \
  --optim adam \
  --accumulation_steps 4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --tau_rgb 0.5 \
  --tau_res 0.5 \
  --eval_batch_size 32 \
  --eval_num_threads 8
```

### 3.2 使用 ProGAN 训练集进行训练

```bash
python train.py \
  --name pgc_train_progan \
  --checkpoints_dir checkpoints \
  --devices 0 \
  --batch_size 32 \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --lora_dropout 0.1 \
  --real_image_dir <DATA_ROOT>/UniversalFakeDetect/train/0_real \
  --fake_image_dir <DATA_ROOT>/UniversalFakeDetect/train/1_fake \
  --test_root <DATA_ROOT>/UniversalFakeDetect/test \
  --cropSize 224 \
  --niter 100 \
  --lr 5e-5 \
  --optim adam \
  --accumulation_steps 4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --tau_rgb 0.5 \
  --tau_res 0.5 \
  --eval_batch_size 32 \
  --eval_num_threads 8
```

### 3.3 ProGAN + SDv1.4 训练集联合训练

```bash
python train.py \
  --name pgc_train_progan_sdv1_4 \
  --checkpoints_dir checkpoints \
  --devices 0 \
  --batch_size 32 \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --lora_dropout 0.1 \
  --real_image_dir \
    <DATA_ROOT>/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature \
    <DATA_ROOT>/UniversalFakeDetect/train/0_real \
  --fake_image_dir \
    <DATA_ROOT>/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai \
    <DATA_ROOT>/UniversalFakeDetect/train/1_fake \
  --test_root <DATA_ROOT>/AIGI/test \
  --cropSize 224 \
  --niter 100 \
  --lr 5e-5 \
  --optim adam \
  --accumulation_steps 4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --tau_rgb 0.5 \
  --tau_res 0.5 \
  --eval_batch_size 32 \
  --eval_num_threads 8
```

## 📊 4. 测试指令 (Evaluation Commands)

我们在不同的 Benchmarks 上均提供相应的测试脚本：

### 4.1 GenImage 基准测试

```bash
python test.py \
  --name test_GenImage \
  --checkpoint <PROJECT_ROOT>/pretrained_model/PGC_train_sdv1_4_ckpt.pth \
  --test_root <DATA_ROOT>/GenImage/test \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --devices 0 \
  --batch_size 32
```

### 4.2 CommGen15 基准测试

```bash
python test.py \
  --name test_CommGen15 \
  --checkpoint <PROJECT_ROOT>/pretrained_model/PGC_train_sdv1_4_ckpt.pth \
  --test_root <DATA_ROOT>/CommGen15 \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --devices 0 \
  --batch_size 32
```

### 4.3 UniversalFakeDetect 基准测试

```bash
python test.py \
  --name test_UniversalFakeDetect \
  --checkpoint <PROJECT_ROOT>/pretrained_model/PGC_train_progan_ckpt.pth \
  --test_root <DATA_ROOT>/UniversalFakeDetect/test \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --devices 0 \
  --batch_size 32
```

### 4.4 AIGI 基准测试

```bash
python test.py \
  --name test_AIGI \
  --checkpoint <PROJECT_ROOT>/pretrained_model/PGC_train_progan_sdv1_4_ckpt.pth \
  --test_root <DATA_ROOT>/AIGI/test \
  --dino_variant dinov2-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --lora_rank 8 \
  --lora_alpha 1.0 \
  --devices 0 \
  --batch_size 32
```


## ✍️ 引用
```
@inproceedings{zhou2026pgc,
  title={PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection},
  author={Zhou, Xiaoyu and Fei, Jianwei and Yu, Peipeng and Xie, Jingchang and Cheng, Chong and Xia, Zhihua},
  journal={arXiv preprint arXiv:2605.21207},
  year={2026}
}
```