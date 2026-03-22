# LLM Geliştiren LLM

Bir LLM'in başka bir LLM'i eğitmesini sağlayan otomatik geliştirme pipeline'ı.

## Mimari

```
Developer LLM (Grok/Groq API)
  ↓ CoT Data Generation, Judge, Experiment Design
Target LLM (Qwen3-8B, Kaggle T4 GPU)
  ↓ SFT → GRPO RL → Rejection Sampling → Alignment GRPO
Evaluation (LiveBench)
  ↓ Regression Check
HF Hub (Adapter Upload) + GitHub (Memory Bank)
```

## Kurulum

### 1. Kaggle Secrets
Kaggle Settings → Secrets kısmına ekleyin:
- `HF_TOKEN` — Hugging Face token
- `GITHUB_PAT` — GitHub Personal Access Token
- `XAI_API_KEY` — X.AI (Grok) API key
- `GROQ_API_KEY` — Groq API key (opsiyonel yedek)
- `CEREBRAS_API_KEY` — Cerebras API key (opsiyonel yedek)

### 2. Kaggle Notebook
1. Yeni notebook oluşturun (GPU T4 x2)
2. `kaggle_cells.py` dosyasındaki her `CELL` bölümünü ayrı hücreye kopyalayın
3. Sırayla çalıştırın

### 3. Alternatif: Tek Komut
```python
!git clone https://{GITHUB_PAT}@github.com/aalpay1968-eng/LLM-DEV.git /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")
!python /kaggle/working/repo/main_notebook.py
```

## Dosya Yapısı

```
LLM-DEV/
├── main_notebook.py          # Ana giriş noktası
├── kaggle_cells.py           # Kaggle hücre rehberi
├── scripts/
│   ├── config.py             # Sabitler ve API havuzu
│   ├── memory_bank.py        # Memory Bank yönetimi
│   ├── developer_llm.py      # Developer LLM (ReAct)
│   ├── rewards.py            # GRPO ödül fonksiyonları
│   ├── training_phases.py    # 4 eğitim aşaması
│   ├── evaluation.py         # LiveBench değerlendirme
│   └── utils.py              # Kaydetme, yedekleme
├── memory/                   # Memory Bank (proje durumu)
│   ├── projectbrief.md
│   ├── techContext.md
│   ├── systemPatterns.md
│   ├── activeContext.md
│   ├── progress.md
│   └── decisionLog.md
├── results/                  # Deney sonuçları (JSON)
└── logs/                     # Eğitim logları (CSV)
```

## Eğitim Aşamaları

| Aşama | Açıklama | Hedef |
|---|---|---|
| Phase 1 | Cold Start SFT | `<think>/<answer>` format öğretme |
| Phase 2 | GRPO RL | Reasoning yeteneğini geliştirme |
| Phase 3 | Rejection Sampling | Veri kalitesini artırma |
| Phase 4 | Alignment GRPO | Güvenli ve yardımcı davranış |

## Lisans
MIT
