# scripts/memory_bank.py -- Memory Bank yonetimi
import os
from datetime import datetime, timezone


def memory_bank_yukle(repo_dir: str) -> dict:
    dosyalar = {
        "clinerules": ".clinerules",
        "projectbrief": "memory/projectbrief.md",
        "techContext": "memory/techContext.md",
        "systemPatterns": "memory/systemPatterns.md",
        "activeContext": "memory/activeContext.md",
        "progress": "memory/progress.md",
        "decisionLog": "memory/decisionLog.md",
    }
    bank = {}
    for anahtar, yol in dosyalar.items():
        tam_yol = os.path.join(repo_dir, yol)
        if os.path.exists(tam_yol):
            with open(tam_yol, encoding="utf-8") as f:
                bank[anahtar] = f.read()
        else:
            bank[anahtar] = f"[{yol} henuz olusturulmadi]"
    return bank


def memory_bank_sistem_prompt_olustur(bank: dict) -> str:
    return f"""
Sen Antigravity projesinin bas orkestrator AI'isin.
Asagidaki Memory Bank'i oku, ozumse ve gorevlerini buna gore yonet.

=== PROJE TANIMI ===
{bank['projectbrief']}

=== TEKNOLOJI KISITLARI ===
{bank['techContext']}

=== ORKESTRASYON KURALLARI ===
{bank['clinerules']}

=== GUNCEL DURUM ===
{bank['activeContext']}

=== MIMARI KARARLAR GECMISI ===
{bank['decisionLog']}

CALISMA YONTEMIN:
Her eylemden once dusunce zinciri yaz, sonra JSON komut ver.
SADECE gecerli JSON dondur -- asla ham metin dondurme.
"""


def decision_log_guncelle(repo_dir: str, karar: str, gerekce: str):
    yol = os.path.join(repo_dir, "memory/decisionLog.md")
    yeni_satir = f"""
## {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Karar:** {karar}
**Gerekce:** {gerekce}
---
"""
    mevcut = ""
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as f:
            mevcut = f.read()
    if "# decisionLog.md" not in mevcut:
        mevcut = "# decisionLog.md\n## Mimari Karar Gunlugu\n\n"
    with open(yol, "w", encoding="utf-8") as f:
        f.write(mevcut + yeni_satir)
