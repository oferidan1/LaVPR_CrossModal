# LaVPR: Benchmarking Language and Vision for Place Recognition

Official implementation of **LaVPR**, a comprehensive framework for bridging natural language and computer vision in the context of Visual Place Recognition (VPR).

---

## 🌟 Key Contributions

* **LaVPR Benchmark:** A massive, curated dataset extending standard VPR benchmarks with over **650,000 aligned natural language descriptions**.
* **Multi-Modal Models:** Two distinct architectural approaches:
1. **Multi-Modal Fusion:** Dynamic weighting of image and text features.
2. **Multi-Modal Alignment:** Cross-modal embedding alignment achieving State-of-the-Art (SOTA) performance.


* **Comprehensive Evaluation:** Support for image-only, text-only, and various fusion-based retrieval modes.

---

## 🛠 Setup

### Environment

This codebase has been tested with **PyTorch 2.9.0**, **CUDA 12.6**, and **Xformers**.

```bash
# Create and activate your environment (optional but recommended)
conda create -n lavpr python=3.12
conda activate lavpr

# Install dependencies
pip install -r requirements.txt

```

---

## 📊 Dataset Preparation

To reproduce our results, download the following datasets:

| Dataset | Purpose | Link |
| --- | --- | --- |
| **GSV-Cities** | Training (Source) | [Download](https://github.com/amaralibey/gsv-cities) |
| **MSLS** | Evaluation | [Download](https://github.com/FrederikWarburg/mapillary_sls) |
| **LaVPR** | Text descriptions | Extract: datasets/descriptions.zip to: datasets/descriptions|
| **LaVPR MSLS-Blur**| Blur augmentation (Will vbe provided upon paper acceptance) | Copy folder: datasets/msls_subsets/query_blur to: msls/val dataset location|
| **LaVPR MSLS-Weather** | Weather augmentation (Will be provided upon paper acceptance) | Copy folder: datasets/msls_subsets/query_weather to: msls/val dataset location|

---

## 🚀 Training

Training on **GSV-Cities** for 10 epochs takes approximately **10 hours** on a single NVIDIA RTX 3090.

```

### 2. Image-Text Alignment Model (Cross-Modal)

```bash
python train.py --cross_modal=2 \
                --model_name=Salesforce/blip-itm-base-coco \
                --embeds_dim=256 \
                --image_size=384 \
                --loss_name=MultiSimilarityLossCM \
                --is_trainable_text_encoder=1 \
                --lora_all_linear=1 \
                --lora_r=64 \
                --train_csv=datasets/descriptions/gsv_cities_descriptions.csv \
                --image_root=PATH_TO_GSV_CITIES_DATASET_LOCATON \
                --val_csv=datasets/descriptions/pitts30k_val_800_queries.csv \
                --val_image_root=PATH_TO_PITTS30K_VAL_DATASET_LOCATON

```

*Checkpoints and logs will be saved automatically to the `/logs` directory.*

---

## 🔍 Evaluation

We provide several evaluation modes to test the versatility of LaVPR.

### 📂 Directory Structure
To ensure the paths are mapped correctly, organize your local dataset as follows:

```text
data/
└── amstertime/
    └── test/               <-- image_root
        ├── database/       <-- database_folder
        └── queries/        <-- queries_folder
```

```text
datasets/
└── descriptions/    
    amstertime_descriptions.csv              <-- amstertime descriptions texts
    amstertime_descriptions_subset.csv       <-- amstertime descriptions subset texts
    gsv_cities_descriptions.csv              <-- gsv cities descriptions texts
    msls_challenge_descriptions.csv          <-- msls challenge descriptions texts
    msls_val_descriptions.csv                <-- msls val descriptions texts
    msls_val_descriptions_blur.csv           <-- msls val descriptions blur texts
    msls_val_descriptions_weather.csv        <-- msls val descriptions weather texts
    pitts30k_test_descriptions.csv           <-- pitts30k test descriptions texts
    pitts30k_val_800_queries.csv             <-- pitts30k val 800 queries texts
    pitts30k_val_descriptions.csv            <-- pitts30k val descriptions texts
   
```

--Models

python eval_vpr.py --vpr_dim=512 --vpr_model_name=openai/clip-vit-base-patch32  --is_dual_encoder=0 --fusion_type=none 
--cross_modal=1 --image_size=224 --text_dim=512

python eval_vpr.py --vpr_dim=768 --vpr_model_name=google/siglip2-base-patch16-224  --is_dual_encoder=0 --fusion_type=none --cross_modal=1 --image_size=224 --text_dim=768

python eval_vpr.py --vpr_dim=512 --vpr_model_name=EVA02-B-16 --is_dual_encoder=0 --fusion_type=none --cross_modal=1 --image_size=224 --text_dim=512

python train.py --cross_modal=2 --fusion_type=none --vpr_model_name=Salesforce/blip-itm-base-coco --vpr_dim=256 --is_text_pooling=0 --is_image_pooling=0 --image_size=384 --loss_name=MultiSimilarityLossCM --is_trainable_text_encoder=1 --batch_size=20

python train.py --cross_modal=2 --model_name=openai/clip-vit-base-patch32 --embeds_dim=512 --image_size=224 --batch_size=60 --gpu=6

python eval_vpr.py --vpr_dim=256 --image_size=384 --vpr_model_name=Salesforce/blip-itm-base-coco --fusion_type=none --cross_modal=2 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=256 --text_dim=256 --is_trainable_text_encoder=0 --lora_path=LOGS/resnet50/blip_lora_r16_qkv/

python eval_vpr.py --vpr_dim=512 --image_size=224 --vpr_model_name=openai/clip-vit-base-patch32 --fusion_type=none --cross_modal=4 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=512 --is_trainable_text_encoder=1 --lora_path=LOGS/resnet50/clip_lora_03/

python eval_vpr.py --vpr_dim=768 --image_size=224 --vpr_model_name=google/siglip2-base-patch16-224 --fusion_type=none --cross_modal=2 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=768 --text_dim=768 --is_trainable_text_encoder=1 --lora_path=LOGS/resnet50/siglip_lora_02/



| Mode | Command Snippet |
| --- | --- |
| **Cross-Modal** | `python eval_vpr.py --cross_modal=2 --embeds_dim=256 --image_size=384 --model_name=Salesforce/blip-itm-base-coco --lora_path=checkpoints/blip_lora_all_r64` |

---

## ❤️ Acknowledgements

This repository builds upon several excellent open-source projects:

* [MixVPR](https://github.com/amaralibey/MixVPR) - State-of-the-art VPR architecture.
* [GSV-Cities](https://github.com/amaralibey/gsv-cities) - Large-scale VPR dataset.
* [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) - Standardized VPR evaluation framework.

---


