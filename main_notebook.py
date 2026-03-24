#!/usr/bin/env python3
"""
LLM Gelistiren LLM -- Main Notebook Script (Phase 2 GRPO + Live Chat)
=====================================================================
Run as:  !python /kaggle/working/repo/main_notebook.py
"""

import os, sys, json, gc, torch, subprocess, threading, re, time
from datetime import datetime, timezone

# =====================================================
# CELL 1: Environment Setup
# =====================================================
print("=" * 60)
print("CELL 1: Environment Setup")
print("=" * 60)

# Kaggle secrets -> env vars
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
    print("[OK] Kaggle secrets loaded.")
except ImportError:
    print("[INFO] No kaggle_secrets, using env vars.")

# W&B setup
wandb_key = os.environ.get("WANDB_API_KEY", "")
if wandb_key:
    try:
        import wandb
        wandb.login(key=wandb_key)
        wandb.init(project="llm-dev", name="phase2-grpo",
                   tags=["grpo", "qwen3-8b"])
        print("[OK] W&B initialized.")
    except Exception as e:
        print(f"[WARN] W&B init failed: {e}")

# Package installs
PACKAGES = [
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
    "huggingface_hub", "trl>=0.12", "gradio>=4.0", "openai",
    "wandb", "matplotlib",
]
for pkg in PACKAGES:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "--no-deps" if "unsloth" in pkg else "-q", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"[WARN] {pkg} install failed, continuing.")

print("[OK] Packages ready.")


# =====================================================
# CELL 2: Project Structure & Git Clone
# =====================================================
print("\n" + "=" * 60)
print("CELL 2: Project Structure & Git Clone")
print("=" * 60)

WORKING_DIR = "/kaggle/working"
REPO_DIR = f"{WORKING_DIR}/repo"
OUTPUTS_DIR = f"{WORKING_DIR}/outputs"
os.environ["WORKING_DIR"] = WORKING_DIR
os.environ["REPO_DIR"] = REPO_DIR
os.environ["OUTPUTS_DIR"] = OUTPUTS_DIR

from scripts.config import (GITHUB_USER, GITHUB_REPO,
                             GIT_USER_EMAIL, GIT_USER_NAME)

GH_PAT = os.environ.get("GITHUB_PAT", "")
REPO_URL = (
    f"https://{GH_PAT}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
    if GH_PAT
    else f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
)

if not os.path.exists(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                   check=False, capture_output=True)
    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR, exist_ok=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"],
                   check=False, capture_output=True)

subprocess.run(["git", "-C", REPO_DIR, "config",
                "user.email", GIT_USER_EMAIL],
               check=False, capture_output=True)
subprocess.run(["git", "-C", REPO_DIR, "config",
                "user.name", GIT_USER_NAME],
               check=False, capture_output=True)

for d in ["memory", "results", "logs", "scripts"]:
    os.makedirs(f"{REPO_DIR}/{d}", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
sys.path.insert(0, REPO_DIR)
print(f"[OK] Repo: {REPO_DIR}")


# =====================================================
# CELL 3: Memory Bank & Checkpoint
# =====================================================
print("\n" + "=" * 60)
print("CELL 3: Memory Bank & Checkpoint")
print("=" * 60)

from scripts.memory_bank import load_memory_bank, format_memory_summary

memory = load_memory_bank(REPO_DIR)
print(format_memory_summary(memory))

from scripts.config import HF_USER, HF_CHAT_REPO
from huggingface_hub import HfApi, snapshot_download

hf_token = os.environ.get("HF_TOKEN", "")
onceki_adapter = None

if hf_token:
    try:
        api = HfApi(token=hf_token)
        info = api.repo_info(
            repo_id=f"{HF_USER}/{HF_CHAT_REPO}", token=hf_token)
        onceki_adapter = snapshot_download(
            repo_id=f"{HF_USER}/{HF_CHAT_REPO}",
            local_dir=f"{WORKING_DIR}/prev_adapter",
            token=hf_token,
        )
        print(f"[OK] Previous adapter loaded: {onceki_adapter}")
    except Exception as e:
        print(f"[INFO] No previous adapter: {e}")


# =====================================================
# CELL 4: Model Loading
# =====================================================
print("\n" + "=" * 60)
print("CELL 4: Model Loading")
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
    print("[OK] Previous LoRA adapter merged.")

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
print(f"[OK] Model loaded: {base_model_name}")
print(f"   Trainable: {trainable:,} / {total:,} "
      f"({100*trainable/total:.2f}%)")
print(f"   VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =====================================================
# CELL 5: Developer LLM Connection
# =====================================================
print("\n" + "=" * 60)
print("CELL 5: Developer LLM Connection")
print("=" * 60)

from scripts.developer_llm import create_dev_client

dev_client, dev_model_name = create_dev_client()
if dev_client:
    print(f"[OK] Developer LLM: {dev_model_name}")
else:
    print("[WARN] Dev LLM not available, offline mode.")


# =====================================================
# CELL 6: Live Chat Interface (Gradio - Background)
# =====================================================
print("\n" + "=" * 60)
print("CELL 6: Live Chat Interface (Background)")
print("=" * 60)

# --- Intelligence scoring helpers ---
_chat_scores = []


def score_response(response_text):
    """Score a model response for reasoning quality."""
    score = 0.0
    details = {}

    # 1. Format check: <think>...</think><answer>...</answer>
    has_think = bool(re.search(r"<think>.*?</think>", response_text,
                               re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response_text,
                                re.DOTALL))
    details["format"] = 1.0 if (has_think and has_answer) else 0.0
    score += details["format"] * 0.3

    # 2. Reasoning depth (think block length)
    think_match = re.search(r"<think>(.*?)</think>", response_text,
                            re.DOTALL)
    if think_match:
        think_len = len(think_match.group(1).strip().split())
        details["reasoning_depth"] = min(think_len / 50.0, 1.0)
    else:
        details["reasoning_depth"] = 0.0
    score += details["reasoning_depth"] * 0.3

    # 3. Answer clarity (answer block conciseness)
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text,
                             re.DOTALL)
    if answer_match:
        answer_len = len(answer_match.group(1).strip().split())
        details["answer_clarity"] = min(answer_len / 20.0, 1.0)
    else:
        details["answer_clarity"] = 0.0
    score += details["answer_clarity"] * 0.2

    # 4. Language consistency (penalize mixed TR/EN)
    tr_words = len(re.findall(
        r'\b(ve|ile|icin|ama|cunku)\b', response_text, re.I))
    en_words = len(re.findall(
        r'\b(and|with|for|but|because)\b', response_text, re.I))
    mixed = tr_words > 2 and en_words > 2
    details["language_consistency"] = 0.0 if mixed else 1.0
    score += details["language_consistency"] * 0.2

    return round(score, 3), details


def chat_fn(message, history):
    """Chat function with intelligence scoring."""
    from unsloth import FastLanguageModel as FLM
    FLM.for_inference(model)

    sys_prompt = (
        "Dusuncelerini <think></think> etiketleri arasinda yaz. "
        "Nihai cevabini <answer></answer> icinde ver."
    )
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True)

    # Score the response
    score, details = score_response(response)
    _chat_scores.append({"q": message, "score": score,
                         "details": details})

    score_bar = "#" * int(score * 20)
    score_display = (
        f"\n\n---\n"
        f"**Intelligence Score: {score:.1%}** [{score_bar}]\n"
        f"- Format: {details['format']:.0%} | "
        f"Reasoning: {details['reasoning_depth']:.0%} | "
        f"Clarity: {details['answer_clarity']:.0%} | "
        f"Lang: {details['language_consistency']:.0%}\n"
        f"- Avg Score ({len(_chat_scores)} msgs): "
        f"{sum(s['score'] for s in _chat_scores)/len(_chat_scores):.1%}"
    )

    return response + score_display


def launch_gradio_thread():
    """Start Gradio in a background thread."""
    try:
        import gradio as gr

        demo = gr.ChatInterface(
            fn=chat_fn,
            title="LLM Dev -- Live Intelligence Chat",
            description=(
                f"Model: {base_model_name} | "
                f"LoRA r=32 | Phase 2 GRPO Training Active\n\n"
                f"Each response is scored for reasoning quality. "
                f"Watch the Intelligence Score improve during training!"
            ),
            examples=[
                "5 + 3 * 2 kactir?",
                "Python'da fibonacci sayilarini hesapla.",
                "TCP ve UDP arasindaki fark nedir?",
                "x^2 - 5x + 6 = 0 denkleminin kokleri nedir?",
                "Newton'un 3. yasasini acikla.",
            ],
        )
        demo.launch(share=True, quiet=True)
        print("[OK] Gradio chat launched (share=True).")
    except Exception as e:
        print(f"[WARN] Gradio failed: {e}")


# Start Gradio in background before training
gradio_thread = threading.Thread(target=launch_gradio_thread, daemon=True)
gradio_thread.start()
print("[OK] Gradio chat starting in background thread...")
time.sleep(3)  # Give it time to initialize


# =====================================================
# CELL 7: Training Pipeline -- Phase 1 + Phase 2
# =====================================================
print("\n" + "=" * 60)
print("CELL 7: Training Pipeline")
print("=" * 60)

from scripts.training_phases import phase1_cold_start_sft, phase2_grpo_rl

sonuclar = []
aktif_asama = memory.get("activeContext", {}).get("asama", "phase1")

# --- Phase 1: Cold Start SFT ---
if aktif_asama in ("phase1", "setup"):
    print("\n>> Phase 1: Cold Start SFT starting...")
    sonuc1 = phase1_cold_start_sft(
        model=model,
        tokenizer=tokenizer,
        dev_client=dev_client,
        dev_model=dev_model_name,
    )
    sonuclar.append(sonuc1)
    gc.collect()
    torch.cuda.empty_cache()
    print("[OK] Phase 1 completed.")

# --- Phase 2: GRPO RL (Auto-enabled) ---
print("\n>> Phase 2: GRPO Reinforcement Learning starting...")
sonuc2 = phase2_grpo_rl(
    model=model,
    tokenizer=tokenizer,
    dev_client=dev_client,
    dev_model=dev_model_name,
)
sonuclar.append(sonuc2)
gc.collect()
torch.cuda.empty_cache()
print("[OK] Phase 2 GRPO completed.")


# =====================================================
# CELL 8: Save Results & HF Upload
# =====================================================
print("\n" + "=" * 60)
print("CELL 8: Save Results")
print("=" * 60)

from scripts.utils import seans_sonu_kaydet

seans_sonu_kaydet(sonuclar, f"{HF_USER}/{HF_CHAT_REPO}")
print("[OK] Results saved to GitHub and HF Hub.")


# =====================================================
# CELL 9: Intelligence Report
# =====================================================
print("\n" + "=" * 60)
print("CELL 9: Intelligence Report")
print("=" * 60)

if _chat_scores:
    avg = sum(s["score"] for s in _chat_scores) / len(_chat_scores)
    print(f"Chat interactions: {len(_chat_scores)}")
    print(f"Average Intelligence Score: {avg:.1%}")
    print("Per-message scores:")
    for i, s in enumerate(_chat_scores):
        print(f"  [{i+1}] {s['score']:.1%} - {s['q'][:50]}...")
else:
    print("No chat interactions recorded.")


# =====================================================
# CELL 10: Post-Training Chat (Keep Alive)
# =====================================================
print("\n" + "=" * 60)
print("CELL 10: Post-Training Chat (Model Ready)")
print("=" * 60)
print("Training complete! The Gradio chat is still active.")
print("You can now test the GRPO-trained model's reasoning.")
print("The share link above will remain active.")

# Keep the process alive for chat
try:
    while gradio_thread.is_alive():
        time.sleep(30)
except KeyboardInterrupt:
    pass

print("\n" + "=" * 60)
print("LLM Dev Pipeline -- Session Complete!")
print("=" * 60)
