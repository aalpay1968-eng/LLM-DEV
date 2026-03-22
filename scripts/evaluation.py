# scripts/evaluation.py — LiveBench ve regresyon kontrolü
import os, json, torch


WORKING_DIR = os.environ.get("WORKING_DIR", "/kaggle/working")
MAX_SEQ = int(os.environ.get("MAX_SEQ", "4096"))


def livebench_kostu(model, tokenizer) -> dict:
    import subprocess
    from unsloth import FastLanguageModel

    model.eval()
    FastLanguageModel.for_inference(model)
    temp_adapter = f"{WORKING_DIR}/temp_eval_adapter"
    model.save_pretrained(temp_adapter)
    tokenizer.save_pretrained(temp_adapter)

    sonuclar = {}
    try:
        kategoriler = ["math", "coding", "reasoning", "instruction_following"]
        for kategori in kategoriler:
            proc = subprocess.run([
                "python", "-m", "livebench.run_livebench",
                "--model-path", temp_adapter,
                "--bench-name", "live_bench",
                "--question-begin", "0", "--question-end", "20",
                "--task-name", kategori,
                "--output-file", f"{WORKING_DIR}/livebench_{kategori}.json",
            ], capture_output=True, text=True, timeout=300)
            dosya = f"{WORKING_DIR}/livebench_{kategori}.json"
            if os.path.exists(dosya):
                with open(dosya) as f:
                    veri = json.load(f)
                sonuclar[f"livebench_{kategori}"] = veri.get("accuracy", 0.0)
            else:
                sonuclar[f"livebench_{kategori}"] = None

        gecerli = [v for v in sonuclar.values() if v is not None]
        sonuclar["livebench_ortalama"] = sum(gecerli) / len(gecerli) if gecerli else 0.0
    except FileNotFoundError:
        # Yedek: arc_challenge
        try:
            from lm_eval.evaluator import simple_evaluate
            from lm_eval.models.huggingface import HFLM
            with torch.no_grad():
                sarici = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=2, max_length=MAX_SEQ)
                r = simple_evaluate(model=sarici, tasks=["arc_challenge"], num_fewshot=25, limit=50)
            sonuclar["arc_challenge_50"] = r["results"]["arc_challenge"]["acc,none"]
            sonuclar["livebench_ortalama"] = sonuclar["arc_challenge_50"]
        except Exception:
            sonuclar["livebench_ortalama"] = 0.0
    finally:
        FastLanguageModel.for_training(model)
        model.train()
        torch.cuda.empty_cache()
        import shutil
        if os.path.exists(temp_adapter):
            shutil.rmtree(temp_adapter, ignore_errors=True)
    return sonuclar


def regresyon_kontrolu(yeni, baz, esik=0.02):
    sorunlar = [
        f"REGRESYON [{m}]: {baz[m]:.3f} → {v:.3f}"
        for m, v in yeni.items()
        if m in baz and v is not None and baz[m] - v > esik
    ]
    if sorunlar:
        for s in sorunlar:
            print(f"❌ {s}")
        print("⛔ Deployment durduruldu.")
        return False
    print("✅ Regresyon yok.")
    return True
