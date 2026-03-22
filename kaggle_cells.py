# LLM Geliştiren LLM — Kaggle Notebook Hücreleri
# ================================================
# Bu dosya Kaggle'da kopyala-yapıştır kullanım içindir.
# Her "# CELL" bölümünü ayrı bir Kaggle hücresine yapıştırın.

# ─────────────────────────────────────────────
# CELL 1: Paket Kurulumu (GPU T4 x2 seçili olmalı)
# ─────────────────────────────────────────────
"""
%%capture
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q huggingface_hub trl>=0.12 gradio>=4.0 openai distilabel livebench
"""

# ─────────────────────────────────────────────
# CELL 2: Secrets & Repo Klonu
# ─────────────────────────────────────────────
"""
import os, subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for key in ["HF_TOKEN", "GROQ_API_KEY", "GITHUB_PAT", "XAI_API_KEY", "CEREBRAS_API_KEY"]:
    try:
        val = secrets.get_secret(key)
        if val: os.environ[key] = val
    except: pass

REPO_DIR = "/kaggle/working/repo"
GH_PAT = os.environ.get("GITHUB_PAT", "")
REPO_URL = f"https://{GH_PAT}@github.com/aalpay1968-eng/LLM-DEV.git" if GH_PAT else "https://github.com/aalpay1968-eng/LLM-DEV.git"

if not os.path.exists(f"{REPO_DIR}/.git"):
    !git clone --depth 1 {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull --ff-only

!git -C {REPO_DIR} config user.email "aalpay1968@gmail.com"
!git -C {REPO_DIR} config user.name "aalpay1968-eng"

import sys
sys.path.insert(0, REPO_DIR)
os.environ["WORKING_DIR"] = "/kaggle/working"
os.environ["REPO_DIR"] = REPO_DIR
os.environ["OUTPUTS_DIR"] = "/kaggle/working/outputs"
os.makedirs("/kaggle/working/outputs", exist_ok=True)
print("✅ Repo klonlandı, path ayarlandı.")
"""

# ─────────────────────────────────────────────
# CELL 3: Model Yükleme
# ─────────────────────────────────────────────
"""
import torch
from unsloth import FastLanguageModel

MAX_SEQ = 4096
os.environ["MAX_SEQ"] = str(MAX_SEQ)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
    trust_remote_code=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"✅ Model: Qwen3-8B | Eğitilebilir: {trainable:,}/{total_p:,} ({100*trainable/total_p:.2f}%)")
print(f"   VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
"""

# ─────────────────────────────────────────────
# CELL 4: Developer LLM Bağlantısı
# ─────────────────────────────────────────────
"""
from scripts.developer_llm import create_dev_client

dev_client, dev_model_name = create_dev_client()
if dev_client:
    print(f"✅ Developer LLM: {dev_model_name}")
else:
    print("⚠️ Developer LLM bağlantısı yok, offline mod.")
"""

# ─────────────────────────────────────────────
# CELL 5: Phase 1 — Cold Start SFT
# ─────────────────────────────────────────────
"""
import gc
from scripts.training_phases import phase1_cold_start_sft

sonuc1 = phase1_cold_start_sft(
    model=model, tokenizer=tokenizer,
    dev_client=dev_client, dev_model=dev_model_name,
)
print(f"✅ Phase 1 tamamlandı. LiveBench: {sonuc1.get('livebench_ortalama', 'N/A')}")
gc.collect(); torch.cuda.empty_cache()
"""

# ─────────────────────────────────────────────
# CELL 6: Phase 2 — GRPO RL (opsiyonel, daha fazla GPU gerekir)
# ─────────────────────────────────────────────
"""
from scripts.training_phases import phase2_grpo_rl

sonuc2 = phase2_grpo_rl(
    model=model, tokenizer=tokenizer,
    dev_client=dev_client, dev_model=dev_model_name,
)
print(f"✅ Phase 2 tamamlandı. LiveBench: {sonuc2.get('livebench_ortalama', 'N/A')}")
gc.collect(); torch.cuda.empty_cache()
"""

# ─────────────────────────────────────────────
# CELL 7: Sonuçları Kaydet & HF Hub Upload
# ─────────────────────────────────────────────
"""
from scripts.utils import seans_sonu_kaydet
from scripts.config import HF_USER, HF_CHAT_REPO

sonuclar = [sonuc1]  # sonuc2 varsa ekle
seans_sonu_kaydet(sonuclar, f"{HF_USER}/{HF_CHAT_REPO}")
print("✅ GitHub & HF Hub'a yedekleme tamamlandı.")
"""

# ─────────────────────────────────────────────
# CELL 8: Chat Arayüzü (Gradio)
# ─────────────────────────────────────────────
"""
import gradio as gr
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)

def chat_fn(message, history):
    messages = [{"role":"system","content":"Düşünme sürecini <think></think> içinde, cevabını <answer></answer> içinde ver."}]
    for h in history:
        messages.append({"role":"user","content":h[0]})
        if h[1]: messages.append({"role":"assistant","content":h[1]})
    messages.append({"role":"user","content":message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95, do_sample=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

demo = gr.ChatInterface(fn=chat_fn, title="🧠 LLM Geliştiren LLM")
demo.launch(share=True)
"""
