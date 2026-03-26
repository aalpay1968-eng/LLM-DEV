#!/usr/bin/env python3
"""
LLM Gelistiren LLM -- Main Notebook Script (Phase 2 GRPO + Live Chat)
=====================================================================
Run as:  !python /kaggle/working/repo/main_notebook.py

Defensive version: every cell wrapped in try/except with file-based logging.
"""

import os, sys, json, gc, traceback, time

# =====================================================
# ERROR LOGGING -- always visible in kernel output
# =====================================================
LOG_FILE = "/kaggle/working/llm_dev_debug.log"

def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def log_exception(cell_name):
    tb = traceback.format_exc()
    log(f"{cell_name} FAILED:\n{tb}", level="ERROR")


# =====================================================
# CELL 1: Environment Setup
# =====================================================
log("=" * 60)
log("CELL 1: Environment Setup")

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
    log("Kaggle secrets loaded.")
except ImportError:
    log("No kaggle_secrets, using env vars.")

# W&B setup (optional, don't crash if it fails)
try:
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        import wandb
        wandb.login(key=wandb_key)
        wandb.init(project="llm-dev", name="phase2-grpo",
                   tags=["grpo", "qwen3-8b"], resume="allow")
        log("W&B initialized.")
except Exception as e:
    log(f"W&B init failed (non-fatal): {e}", "WARN")

# Package installs
import subprocess
PACKAGES = [
    "unsloth_zoo",
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
    "huggingface_hub", "trl>=0.12", "gradio>=4.0", "openai",
    "matplotlib",
]
for pkg in PACKAGES:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        log(f"Package install failed: {pkg}", "WARN")

log("Packages ready.")

import torch
log(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =====================================================
# CELL 2: Project Structure & Git Clone
# =====================================================
log("=" * 60)
log("CELL 2: Project Structure & Git Clone")

WORKING_DIR = "/kaggle/working"
REPO_DIR = f"{WORKING_DIR}/repo"
OUTPUTS_DIR = f"{WORKING_DIR}/outputs"
os.environ["WORKING_DIR"] = WORKING_DIR
os.environ["REPO_DIR"] = REPO_DIR
os.environ["OUTPUTS_DIR"] = OUTPUTS_DIR

GITHUB_USER = "aalpay1968-eng"
GITHUB_REPO = "LLM-DEV"

try:
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

    sys.path.insert(0, REPO_DIR)

    # Git config -- use hardcoded values as fallback
    git_email = "aalpay1968@gmail.com"
    git_name = "aalpay1968-eng"
    try:
        from scripts.config import GIT_USER_EMAIL, GIT_USER_NAME
        git_email = GIT_USER_EMAIL
        git_name = GIT_USER_NAME
    except ImportError:
        log("scripts.config import failed, using defaults", "WARN")

    subprocess.run(["git", "-C", REPO_DIR, "config", "user.email", git_email],
                   check=False, capture_output=True)
    subprocess.run(["git", "-C", REPO_DIR, "config", "user.name", git_name],
                   check=False, capture_output=True)

    for d in ["memory", "results", "logs", "scripts"]:
        os.makedirs(f"{REPO_DIR}/{d}", exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    log(f"Repo: {REPO_DIR}")
except Exception:
    log_exception("CELL 2")


# =====================================================
# CELL 3: Memory Bank & Checkpoint
# =====================================================
log("=" * 60)
log("CELL 3: Memory Bank & Checkpoint")

memory = {}
onceki_adapter = None

try:
    # Try Turkish names first, fall back to English aliases
    try:
        from scripts.memory_bank import memory_bank_yukle, memory_bank_sistem_prompt_olustur
    except ImportError:
        from scripts.memory_bank import load_memory_bank as memory_bank_yukle
        from scripts.memory_bank import format_memory_summary as memory_bank_sistem_prompt_olustur

    memory = memory_bank_yukle(REPO_DIR)
    log("Memory Bank loaded.")
except Exception:
    log_exception("CELL 3 Memory Bank")
    memory = {"clinerules": "", "projectbrief": "", "techContext": "",
              "systemPatterns": "", "activeContext": "phase2",
              "progress": "", "decisionLog": ""}
    log("Using empty memory bank as fallback.", "WARN")

try:
    from huggingface_hub import HfApi, snapshot_download
    hf_token = os.environ.get("HF_TOKEN", "")
    HF_USER = "aalpay1968"
    HF_CHAT_REPO = "target-llm-chat"

    if hf_token:
        try:
            api = HfApi(token=hf_token)
            info = api.repo_info(repo_id=f"{HF_USER}/{HF_CHAT_REPO}", token=hf_token)
            onceki_adapter = snapshot_download(
                repo_id=f"{HF_USER}/{HF_CHAT_REPO}",
                local_dir=f"{WORKING_DIR}/prev_adapter",
                token=hf_token,
            )
            log(f"Previous adapter loaded: {onceki_adapter}")
        except Exception as e:
            log(f"No previous adapter: {e}", "WARN")
except Exception:
    log_exception("CELL 3 Checkpoint")
    HF_USER = "aalpay1968"
    HF_CHAT_REPO = "target-llm-chat"


# =====================================================
# CELL 3.5: GPU Compatibility Check
# =====================================================
log("=" * 60)
log("CELL 3.5: GPU Compatibility Check")

try:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        gpu_arch = f"sm_{cap_major}{cap_minor}"
        log(f"GPU: {gpu_name} (arch: {gpu_arch})")

        if cap_major < 7:
            log(f"{gpu_name} ({gpu_arch}) is older than sm_70. Reinstalling Official PyTorch to restore P100 support...", "WARN")
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y",
                "torch", "torchvision", "torchaudio", "triton"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                "torch==2.5.1+cu118", "torchvision==0.20.1+cu118", "torchaudio==2.5.1+cu118", "triton",
                "--index-url", "https://download.pytorch.org/whl/cu118",
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("PyTorch restored for P100 compatibility.")
        else:
            log(f"GPU {gpu_name} ({gpu_arch}) is fully compatible.")
    else:
        log("No CUDA GPU available!", "ERROR")
        sys.exit(1)
except Exception:
    log_exception("CELL 3.5")
    log("GPU check failed, attempting to continue...", "WARN")


# =====================================================
# CELL 4: Model Loading
# =====================================================
log("=" * 60)
log("CELL 4: Model Loading")

model = None
tokenizer = None
base_model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

try:
    from unsloth import FastLanguageModel

    MAX_SEQ = 4096
    os.environ["MAX_SEQ"] = str(MAX_SEQ)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )

    if onceki_adapter and os.path.exists(str(onceki_adapter)):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, onceki_adapter)
            log("Previous LoRA adapter merged.")
        except Exception as e:
            log(f"LoRA merge failed (non-fatal): {e}", "WARN")

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
    log(f"Model loaded: {base_model_name}")
    log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    log(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
except Exception:
    log_exception("CELL 4")
    log("Model loading failed! Cannot continue training.", "ERROR")


# =====================================================
# CELL 5: Developer LLM Connection
# =====================================================
log("=" * 60)
log("CELL 5: Developer LLM Connection")

dev_client = None
dev_model_name = None

try:
    from scripts.developer_llm import developer_client_olustur
    from scripts.config import DEVELOPER_APIS
    from scripts.memory_bank import memory_bank_sistem_prompt_olustur as _prompt_fn

    dev_client, dev_model_name, dev_extra_params, dev_sistem_prompt = (
        developer_client_olustur(memory, DEVELOPER_APIS, _prompt_fn)
    )
    log(f"Developer LLM: {dev_model_name}")
except Exception as e:
    log(f"Dev LLM not available, offline mode: {e}", "WARN")


# =====================================================
# CELL 6: Live Chat Interface (Gradio - Background)
# =====================================================
log("=" * 60)
log("CELL 6: Live Chat Interface (Background)")

import re, threading

_chat_scores = []

def score_response(response_text):
    score = 0.0
    details = {}
    has_think = bool(re.search(r"<think>.*?</think>", response_text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response_text, re.DOTALL))
    details["format"] = 1.0 if (has_think and has_answer) else 0.0
    score += details["format"] * 0.3
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if think_match:
        think_len = len(think_match.group(1).strip().split())
        details["reasoning_depth"] = min(think_len / 50.0, 1.0)
    else:
        details["reasoning_depth"] = 0.0
    score += details["reasoning_depth"] * 0.3
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if answer_match:
        answer_len = len(answer_match.group(1).strip().split())
        details["answer_clarity"] = min(answer_len / 20.0, 1.0)
    else:
        details["answer_clarity"] = 0.0
    score += details["answer_clarity"] * 0.2
    tr_words = len(re.findall(r'\b(ve|ile|icin|ama|cunku)\b', response_text, re.I))
    en_words = len(re.findall(r'\b(and|with|for|but|because)\b', response_text, re.I))
    mixed = tr_words > 2 and en_words > 2
    details["language_consistency"] = 0.0 if mixed else 1.0
    score += details["language_consistency"] * 0.2
    return round(score, 3), details


def chat_fn(message, history):
    if model is None:
        return "Model yuklenmedi. Lutfen bekleyin."
    try:
        from unsloth import FastLanguageModel as FLM
        FLM.for_inference(model)
    except Exception:
        pass

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
            **inputs, max_new_tokens=1024,
            temperature=0.6, top_p=0.95, do_sample=True,
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    score, details = score_response(response)
    _chat_scores.append({"q": message, "score": score, "details": details})
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


gradio_url_file = "/kaggle/working/gradio_url.txt"

def launch_gradio_thread():
    try:
        import gradio as gr
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="LLM Dev -- Live Intelligence Chat",
            description=(
                f"Model: {base_model_name} | "
                f"LoRA r=32 | Phase 2 GRPO Training Active\n\n"
                f"Each response is scored for reasoning quality."
            ),
            examples=[
                "5 + 3 * 2 kactir?",
                "Python'da fibonacci sayilarini hesapla.",
                "TCP ve UDP arasindaki fark nedir?",
                "x^2 - 5x + 6 = 0 denkleminin kokleri nedir?",
                "Newton'un 3. yasasini acikla.",
            ],
        )
        # Launch and capture the share URL
        app, local_url, share_url = demo.launch(share=True, quiet=True)
        if share_url:
            log(f"GRADIO_URL={share_url}")
            with open(gradio_url_file, "w") as f:
                f.write(share_url)
        else:
            log("Gradio launched but no share URL", "WARN")
    except Exception as e:
        log(f"Gradio failed: {e}", "WARN")


if model is not None:
    gradio_thread = threading.Thread(target=launch_gradio_thread, daemon=True)
    gradio_thread.start()
    log("Gradio chat starting in background thread...")
    time.sleep(5)
else:
    log("Skipping Gradio — model not loaded.", "WARN")


# =====================================================
# CELL 7: Training Pipeline -- Phase 2 GRPO
# =====================================================
log("=" * 60)
log("CELL 7: Training Pipeline")

if model is not None:
    try:
        from scripts.training_phases import phase2_grpo_rl

        sonuclar = []
        _active_ctx = memory.get("activeContext", "")
        aktif_asama = "phase2" if "phase2" in _active_ctx else "phase1"
        has_prev_adapter = onceki_adapter and os.path.exists(str(onceki_adapter))
        if has_prev_adapter:
            log("Previous adapter found, skipping Phase 1.")
            aktif_asama = "phase2"

        # --- Phase 1: Cold Start SFT (only on first run) ---
        if aktif_asama in ("phase1", "setup"):
            log("Phase 1: Cold Start SFT starting...")
            try:
                from scripts.training_phases import phase1_cold_start_sft
                sonuc1 = phase1_cold_start_sft(
                    model=model, tokenizer=tokenizer,
                    dev_client=dev_client, dev_model=dev_model_name,
                )
                sonuclar.append(sonuc1)
                log("Phase 1 completed.")
            except Exception:
                log_exception("Phase 1")
            gc.collect()
            torch.cuda.empty_cache()

        # --- Phase 2: GRPO RL ---
        N_GRPO_ROUNDS = 3
        for grpo_round in range(N_GRPO_ROUNDS):
            log(f"Phase 2: GRPO Round {grpo_round + 1}/{N_GRPO_ROUNDS}...")
            try:
                sonuc2 = phase2_grpo_rl(
                    model=model, tokenizer=tokenizer,
                    dev_client=dev_client, dev_model=dev_model_name,
                )
                sonuclar.append(sonuc2)
                log(f"GRPO Round {grpo_round + 1} completed.")
            except Exception:
                log_exception(f"GRPO Round {grpo_round + 1}")
            gc.collect()
            torch.cuda.empty_cache()

    except Exception:
        log_exception("CELL 7 Training Pipeline")
        sonuclar = []
else:
    log("Skipping training — model not loaded.", "ERROR")
    sonuclar = []


# =====================================================
# CELL 8: Save Results & HF Upload
# =====================================================
log("=" * 60)
log("CELL 8: Save Results")

try:
    if sonuclar:
        from scripts.utils import seans_sonu_kaydet
        seans_sonu_kaydet(sonuclar, f"{HF_USER}/{HF_CHAT_REPO}")
        log("Results saved to GitHub and HF Hub.")
    else:
        log("No results to save.", "WARN")
except Exception:
    log_exception("CELL 8")


# =====================================================
# CELL 9: Intelligence Report
# =====================================================
log("=" * 60)
log("CELL 9: Intelligence Report")

if _chat_scores:
    avg = sum(s["score"] for s in _chat_scores) / len(_chat_scores)
    log(f"Chat interactions: {len(_chat_scores)}")
    log(f"Average Intelligence Score: {avg:.1%}")
else:
    log("No chat interactions recorded.")


# =====================================================
# CELL 10: Post-Training Chat (Keep Alive)
# =====================================================
log("=" * 60)
log("CELL 10: Post-Training Chat (Model Ready)")
log("Training complete! The Gradio chat is still active.")

# Keep the process alive for chat (max 30 minutes post-training)
try:
    end_time = time.time() + 1800  # 30 minutes
    while time.time() < end_time:
        if model is not None and hasattr(gradio_thread, 'is_alive') and gradio_thread.is_alive():
            time.sleep(30)
        else:
            break
except KeyboardInterrupt:
    pass

log("=" * 60)
log("LLM Dev Pipeline -- Session Complete!")
