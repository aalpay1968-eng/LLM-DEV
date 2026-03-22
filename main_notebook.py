#!/usr/bin/env python3
"""
LLM Geliştiren LLM — Ana Notebook Betiği
==========================================
Bu dosya Kaggle notebook'un ana giriş noktasıdır.
Kaggle'da her hücreyi ayrı ayrı çalıştırmak yerine bu dosyayı
tek seferde çalıştırabilirsiniz.

Kullanım (Kaggle notebook hücresinde):
    !python /kaggle/working/repo/main_notebook.py
"""

import os, sys, json, gc, torch, subprocess
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CELL 1: Ortam Kurulumu
# ─────────────────────────────────────────────
print("=" * 60)
print("CELL 1: Ortam Kurulumu")
print("=" * 60)

# Kaggle secrets → env vars
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    KEY_MAP = {
        "HF_TOKEN": "HF_TOKEN",
        "GROQ_API_KEY": "GROQ_API_KEY",
        "GITHUB_PAT": "GITHUB_PAT",
        "XAI_API_KEY": "XAI_API_KEY",
        "CEREBRAS_API_KEY": "CEREBRAS_API_KEY",
        "WANDB_API_KEY": "WANDB_API_KEY",
    }
    for env_name, secret_name in KEY_MAP.items():
        try:
            val = secrets.get_secret(secret_name)
            if val:
                os.environ[env_name] = val
        except Exception:
            pass
    print("✅ Kaggle secrets yüklendi.")
except ImportError:
    print("ℹ️ Kaggle secrets mevcut değil, env vars kullanılıyor.")

# Paket kurulumları
PACKAGES = [
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
    "huggingface_hub", "trl>=0.12", "gradio>=4.0", "openai",
    "livebench", "distilabel[hf-inference-endpoints]",
]
for pkg in PACKAGES:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps"
             if "unsloth" in pkg else "-q", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"⚠️ {pkg} kurulamadı, devam ediliyor.")

print("✅ Paketler hazır.")


# ─────────────────────────────────────────────
# CELL 2: Proje Yapısı & Git Klonu
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 2: Proje Yapısı & Git Klonu")
print("=" * 60)

WORKING_DIR = "/kaggle/working"
REPO_DIR = f"{WORKING_DIR}/repo"
OUTPUTS_DIR = f"{WORKING_DIR}/outputs"
os.environ["WORKING_DIR"] = WORKING_DIR
os.environ["REPO_DIR"] = REPO_DIR
os.environ["OUTPUTS_DIR"] = OUTPUTS_DIR

from scripts.config import GITHUB_USER, GITHUB_REPO, GIT_USER_EMAIL, GIT_USER_NAME

GH_PAT = os.environ.get("GITHUB_PAT", "")
REPO_URL = (f"https://{GH_PAT}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
            if GH_PAT else f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git")

if not os.path.exists(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                   check=False, capture_output=True)
    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR, exist_ok=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"],
                   check=False, capture_output=True)

# Git yapılandırması
subprocess.run(["git", "-C", REPO_DIR, "config", "user.email", GIT_USER_EMAIL],
               check=False, capture_output=True)
subprocess.run(["git", "-C", REPO_DIR, "config", "user.name", GIT_USER_NAME],
               check=False, capture_output=True)

for d in ["memory", "results", "logs", "scripts"]:
    os.makedirs(f"{REPO_DIR}/{d}", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# scripts/ kopyala
sys.path.insert(0, REPO_DIR)
print(f"✅ Repo: {REPO_DIR}")


# ─────────────────────────────────────────────
# CELL 3: Memory Bank & Checkpoint Yükleme
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 3: Memory Bank & Checkpoint")
print("=" * 60)

from scripts.memory_bank import load_memory_bank, format_memory_summary

memory = load_memory_bank(REPO_DIR)
print(format_memory_summary(memory))

# Önceki adapter var mı?
from scripts.config import HF_USER, HF_CHAT_REPO
from huggingface_hub import HfApi, snapshot_download

hf_token = os.environ.get("HF_TOKEN", "")
onceki_adapter = None

if hf_token:
    try:
        api = HfApi(token=hf_token)
        info = api.repo_info(repo_id=f"{HF_USER}/{HF_CHAT_REPO}", token=hf_token)
        onceki_adapter = snapshot_download(
            repo_id=f"{HF_USER}/{HF_CHAT_REPO}",
            local_dir=f"{WORKING_DIR}/prev_adapter",
            token=hf_token,
        )
        print(f"✅ Önceki adapter yüklendi: {onceki_adapter}")
    except Exception as e:
        print(f"ℹ️ Önceki adapter yok ya da erişilemedi: {e}")


# ─────────────────────────────────────────────
# CELL 4: Model Yükleme
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 4: Model Yükleme")
print("=" * 60)

from unsloth import FastLanguageModel

MAX_SEQ = 4096
os.environ["MAX_SEQ"] = str(MAX_SEQ)

base_model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
    trust_remote_code=True,
)

if onceki_adapter and os.path.exists(str(onceki_adapter)):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, onceki_adapter)
    print("✅ Önceki LoRA adapter uygulandı.")

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Model yüklendi: {base_model_name}")
print(f"   Eğitilebilir: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"   VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ─────────────────────────────────────────────
# CELL 5: Developer LLM Bağlantısı
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 5: Developer LLM Bağlantısı")
print("=" * 60)

from scripts.developer_llm import create_dev_client

dev_client, dev_model_name = create_dev_client()
if dev_client:
    print(f"✅ Developer LLM: {dev_model_name}")
else:
    print("⚠️ Developer LLM bağlantısı kurulamadı, offline modda devam.")


# ─────────────────────────────────────────────
# CELL 6: Baz Değerlendirme
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 6: Baz Değerlendirme")
print("=" * 60)

from scripts.evaluation import livebench_kostu
from scripts.utils import yuk_son_baz_skor

baz_skor = yuk_son_baz_skor()
if not baz_skor:
    print("İlk çalıştırma, baz skor ölçülüyor...")
    baz_skor = livebench_kostu(model, tokenizer)
    print(f"Baz skor: {json.dumps(baz_skor, indent=2)}")
else:
    print(f"Mevcut baz skor: {json.dumps(baz_skor, indent=2)}")


# ─────────────────────────────────────────────
# CELL 7: Eğitim Pipeline
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 7: Eğitim Pipeline — Phase 1 Cold Start SFT")
print("=" * 60)

from scripts.training_phases import phase1_cold_start_sft

sonuclar = []

# Aktif aşamayı belirle
aktif_asama = memory.get("activeContext", {}).get("asama", "phase1")

if aktif_asama in ("phase1", "setup"):
    print("▶ Phase 1: Cold Start SFT başlıyor...")
    sonuc1 = phase1_cold_start_sft(
        model=model,
        tokenizer=tokenizer,
        dev_client=dev_client,
        dev_model=dev_model_name,
    )
    sonuclar.append(sonuc1)
    baz_skor.update({k: v for k, v in sonuc1.items() if "livebench" in k and isinstance(v, (int, float))})
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Phase 1 tamamlandı. LiveBench: {sonuc1.get('livebench_ortalama', 'N/A')}")


# ─────────────────────────────────────────────
# CELL 8: Phase 2 GRPO RL (Opsiyonel)
# ─────────────────────────────────────────────
DEVAM_PHASE2 = os.environ.get("DEVAM_PHASE2", "false").lower() == "true"

if DEVAM_PHASE2 or aktif_asama == "phase2":
    print("\n" + "=" * 60)
    print("CELL 8: Phase 2 — GRPO RL")
    print("=" * 60)

    from scripts.training_phases import phase2_grpo_rl

    sonuc2 = phase2_grpo_rl(
        model=model,
        tokenizer=tokenizer,
        dev_client=dev_client,
        dev_model=dev_model_name,
    )
    sonuclar.append(sonuc2)
    baz_skor.update({k: v for k, v in sonuc2.items() if "livebench" in k and isinstance(v, (int, float))})
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Phase 2 tamamlandı. LiveBench: {sonuc2.get('livebench_ortalama', 'N/A')}")


# ─────────────────────────────────────────────
# CELL 9: Sonuçları Kaydet & HF Upload
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 9: Sonuçları Kaydet")
print("=" * 60)

from scripts.utils import seans_sonu_kaydet

seans_sonu_kaydet(sonuclar, f"{HF_USER}/{HF_CHAT_REPO}")
print("✅ Sonuçlar GitHub ve HF Hub'a kaydedildi.")


# ─────────────────────────────────────────────
# CELL 10: Chat Arayüzü (Gradio)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CELL 10: Chat Arayüzü")
print("=" * 60)

try:
    import gradio as gr
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    def chat_fn(message, history):
        messages = [{"role": "system", "content": "Sen yardımcı bir asistansın. Düşünme sürecini <think></think> içinde göster, cevabını <answer></answer> içinde ver."}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🧠 LLM Geliştiren LLM — Chat",
        description=f"Model: {base_model_name} | Adapter: LoRA r=32",
        examples=["Türkiye'nin başkenti neresidir?", "Python'da fibonacci sayılarını hesapla.", "Yapay zekanın geleceği hakkında ne düşünüyorsun?"],
    )
    demo.launch(share=True, quiet=True)
    print("✅ Gradio arayüzü başlatıldı.")

except Exception as e:
    print(f"⚠️ Gradio başlatılamadı: {e}")
    print("Chat arayüzü olmadan devam ediliyor.")


print("\n" + "=" * 60)
print("🎉 LLM Geliştiren LLM — Seans tamamlandı!")
print("=" * 60)
