# scripts/rewards.py -- GRPO odul fonksiyonlari
import re


def odul_format_kontrol(completions, **kwargs):
    """<think>...</think><answer>...</answer> formati var mi?"""
    odullar = []
    for completion in completions:
        metin = completion[0]["content"] if isinstance(completion, list) else completion
        think_var = bool(re.search(r"<think>.*?</think>", metin, re.DOTALL))
        answer_var = bool(re.search(r"<answer>.*?</answer>", metin, re.DOTALL))
        odullar.append(0.5 if (think_var and answer_var) else 0.0)
    return odullar


def odul_matematik_dogruluk(completions, ground_truth=None, **kwargs):
    odullar = []
    for i, completion in enumerate(completions):
        metin = completion[0]["content"] if isinstance(completion, list) else completion
        answer_match = re.search(r"<answer>(.*?)</answer>", metin, re.DOTALL)
        if not answer_match:
            odullar.append(0.0)
            continue
        model_cevap = answer_match.group(1).strip()
        gercek = (ground_truth[i] if ground_truth and i < len(ground_truth) else "")
        odullar.append(1.0 if model_cevap == gercek else 0.0)
    return odullar


def odul_kod_calistir(completions, test_kodu=None, **kwargs):
    import subprocess, tempfile, os
    odullar = []
    for completion in completions:
        metin = completion[0]["content"] if isinstance(completion, list) else completion
        kod_match = re.search(r"```python\n(.*?)```", metin, re.DOTALL)
        if not kod_match:
            odullar.append(0.0)
            continue
        kod = kod_match.group(1)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(kod)
                tmp_path = f.name
            sonuc = subprocess.run(
                ["python", tmp_path], capture_output=True, text=True, timeout=10)
            odullar.append(1.0 if sonuc.returncode == 0 else 0.0)
        except Exception:
            odullar.append(0.0)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    return odullar


def odul_dil_karisikligi_ceza(completions, **kwargs):
    odullar = []
    for completion in completions:
        metin = completion[0]["content"] if isinstance(completion, list) else completion
        tr_kelimeler = len(re.findall(r'\b(ve|ile|icin|ama|cunku)\b', metin, re.I))
        en_kelimeler = len(re.findall(r'\b(and|with|for|but|because)\b', metin, re.I))
        karisik = tr_kelimeler > 2 and en_kelimeler > 2
        odullar.append(-0.3 if karisik else 0.0)
    return odullar


def odul_anac_empati(completions, **kwargs):
    """
    Anaç/Şefkatli LLM için Empati ve Destek Odülü.
    Yanıtın destekleyici, cesaretlendirici kelimelerini içerip içermediğini kontrol eder.
    """
    import re
    SEFKAT_KELIMELERI = [
        "canım", "yanındayım", "hissediyorum", "anlıyorum", "geçecek", 
        "değerlisin", "beraber", "harikasın", "güçlüsün", "üzülme",
        "destek", "merak etme", "korkma", "mutlu", "hata", "öğrenme", "sarılıyorum",
        "kıyamam", "canını sıkma", "gurur"
    ]
    odullar = []
    for completion in completions:
        metin = (completion[0]["content"] if isinstance(completion, list) else completion).lower()
        answer_match = re.search(r"<answer>(.*?)</answer>", metin, re.DOTALL)
        if not answer_match:
            odullar.append(0.0)
            continue
        cevap = answer_match.group(1).strip()
        
        # Kelime bazli empati basarisi (0.1 points per word, max 0.6)
        bulunan = sum(1 for kelime in SEFKAT_KELIMELERI if kelime in cevap)
        odullar.append(min(0.6, bulunan * 0.1))
    return odullar
