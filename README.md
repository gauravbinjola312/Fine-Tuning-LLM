# Fine-Tuning DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset

This project demonstrates how to fine-tune the **DeepSeek-R1-Distill-Llama-8B** model using the **Unsloth** library for efficient training. The workflow leverages **Hugging Face Hub**, **Weights & Biases (W&B)**, and **Kaggle Secrets** for secure training and experiment tracking.  

## 🚀 Features
- Fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) for **faster and memory-efficient training**.  
- Supports **4-bit quantization** to reduce GPU memory usage.  
- **W&B integration** for experiment tracking.  
- Uses **Hugging Face Datasets** for loading and preprocessing data.  
- Optimized for **Chain-of-Thought (COT)** reasoning tasks in the **medical domain**.  

## 📦 Installation

```bash
# Install Unsloth
pip install unsloth  

# Install latest version from GitHub
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git  

# Additional dependencies
pip install torch transformers datasets accelerate trl wandb
```

## 🔑 Setup
Before running the notebook, ensure you have:
1. **Hugging Face Token** stored in Kaggle Secrets as `Hugging_Face_Token`.  
2. **Weights & Biases Token** stored as `wnb`.  

These tokens will be automatically fetched using:  
```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hugging_face_token = user_secrets.get_secret("Hugging_Face_Token")
wnb_token = user_secrets.get_secret("wnb")
```

## ⚙️ Training Configuration
- **Model:** `unsloth/DeepSeek-R1-Distill-Llama-8B`  
- **Max Sequence Length:** 2048 tokens  
- **Quantization:** 4-bit (saves GPU memory)  
- **Precision:** Auto-detect (bfloat16 if supported)  

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## 📊 Experiment Tracking
Training runs are logged to **Weights & Biases** automatically:  

```python
import wandb
wandb.login(key=wnb_token)

run = wandb.init(
    project="Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset",
    job_type="training",
    anonymous="allow"
)
```

## 📂 Dataset
- Loaded using Hugging Face `datasets` library.  
- Dataset should be preprocessed into **prompt–response** format for supervised fine-tuning (SFT).  

Example:
```python
from datasets import load_dataset
dataset = load_dataset("your_dataset_name")
```

## ▶️ Usage
Run the notebook step by step:
1. Install dependencies.  
2. Authenticate with Hugging Face & W&B.  
3. Load dataset.  
4. Fine-tune using `SFTTrainer`.  
5. Evaluate and save model.  

## 💾 Saving & Pushing Model
After fine-tuning, push the model to your Hugging Face Hub:  
```python
model.push_to_hub("your-username/deepseek-medical-finetuned")
tokenizer.push_to_hub("your-username/deepseek-medical-finetuned")
```

## 📌 Requirements
- Python 3.9+  
- CUDA-enabled GPU (recommended for faster training)  
- At least 16GB GPU VRAM (with 4-bit quantization)  
