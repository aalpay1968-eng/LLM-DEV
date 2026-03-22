# scripts/utils.py — Sonuç kaydetme, Memory Bank güncelleme, seans sonu
import os, json, csv, subprocess, time
from datetime import datetime, timezone


WORKING_DIR = os.environ.get("WORKING_DIR", "/kaggle/working")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", f"{WORKING_DIR}/outputs")
REPO_DIR = os.environ.get("REPO_DIR", f"{WORKING_DIR}/repo")


def sonuclari_kaydet(iter_sonuc):
    zaman = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dosya = f"{REPO_DIR}/results/iter_{zaman}.json"
    os.makedirs(os.path.dirname(dosya), exist_ok=True)
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(iter_sonuc, f, indent=2, ensure_ascii=False, default=str)

    log_dosya = f"{REPO_DIR}/logs/training_log.csv"
    os.makedirs(os.path.dirname(log_dosya), exist_ok=True)
    ilk = not os.path.exists(log_dosya)
    with open(log_dosya, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zaman", "deney_id", "asama",
                                           "livebench_ortalama", "deney_aciklama"])
        if ilk:
            w.writeheader()
        w.writerow({
            "zaman": zaman,
            "deney_id": iter_sonuc.get("deney_id", "?"),
            "asama": iter_sonuc.get("asama", "?"),
            "livebench_ortalama": iter_sonuc.get("livebench_ortalama", 0),
            "deney_aciklama": iter_sonuc.get("degisiklik_aciklama", ""),
        })

    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results/", "logs/"],
                       check=True, capture_output=True)
        mesaj = (f"iter: {iter_sonuc.get('deney_id','?')} | "
                 f"asama={iter_sonuc.get('asama','?')} | "
                 f"livebench={iter_sonuc.get('livebench_ortalama', 0):.3f}")
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m", mesaj],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", REPO_DIR, "push"], check=True, capture_output=True)
        print(f"✅ GitHub: {mesaj}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        if "nothing to commit" not in stderr:
            print(f"⚠️ Git: {stderr[:200]}")


def memory_guncelle(son):
    zaman = datetime.now(timezone.utc).isoformat()
    aktif = f"""# activeContext.md
Son güncelleme: {zaman}

## Model Durumu
- Son deney: {son.get('deney_id','?')}
- Mevcut aşama: {son.get('asama','?')}
- LiveBench ortalama: {son.get('livebench_ortalama','N/A')}

## Sonraki Seans
- snapshot_download ile devam et
"""
    progress_dosya = f"{REPO_DIR}/memory/progress.md"
    mevcut = ""
    if os.path.exists(progress_dosya):
        with open(progress_dosya, encoding="utf-8") as f:
            mevcut = f.read()
    if "## İlerleme" not in mevcut:
        mevcut = "# progress.md\n\n## İlerleme\n\n| Zaman | Aşama | Deney | LiveBench |\n|---|---|---|---|\n"
    mevcut += (f"| {zaman[:19]} | {son.get('asama','?')} | "
               f"{son.get('deney_id','?')} | {son.get('livebench_ortalama','N/A')} |\n")

    for yol, icerik in [
        (f"{REPO_DIR}/memory/activeContext.md", aktif),
        (progress_dosya, mevcut),
    ]:
        os.makedirs(os.path.dirname(yol), exist_ok=True)
        with open(yol, "w", encoding="utf-8") as f:
            f.write(icerik)

    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "memory/"], check=True, capture_output=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m",
                         f"memory: {son.get('deney_id','')}"], check=True, capture_output=True)
        subprocess.run(["git", "-C", REPO_DIR, "push"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pass


def yuk_son_baz_skor():
    import glob
    dosyalar = sorted(glob.glob(f"{REPO_DIR}/results/iter_*.json"))
    if not dosyalar:
        return {}
    try:
        with open(dosyalar[-1], encoding="utf-8") as f:
            son = json.load(f)
        return {k: v for k, v in son.items() if "livebench" in k and isinstance(v, float)}
    except Exception:
        return {}


def seans_sonu_kaydet(sonuclar, hf_repo_chat):
    if not sonuclar:
        print("⚠️ Kaydedilecek sonuç yok.")
        return
    son = sonuclar[-1]
    import glob
    from scripts.training_phases import _hf_yukle
    lora_yol = son.get("lora_yol")
    if not lora_yol or not os.path.exists(str(lora_yol)):
        kandidatlar = sorted([
            p for p in [
                f"{WORKING_DIR}/phase1_adapter",
                f"{WORKING_DIR}/phase2_adapter",
                *glob.glob(f"{OUTPUTS_DIR}/checkpoint-*"),
            ] if os.path.exists(p)
        ])
        lora_yol = kandidatlar[-1] if kandidatlar else None
    if lora_yol:
        _hf_yukle(lora_yol, hf_repo_chat)
    else:
        print("⚠️ Adapter bulunamadı.")
    sonuclari_kaydet(son)
    memory_guncelle(son)
    print("✅ Seans sonu yedekleme tamamlandı.")
