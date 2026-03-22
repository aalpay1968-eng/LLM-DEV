# techContext.md
## Teknoloji Yığını Kısıtları

### Donanım
- Platform: Kaggle T4 (16 GB VRAM) veya T4 x2 (DDP)
- Kaggle hesabı: adilalpay (telefon doğrulamalı, GPU aktif)
- GPU kotası: Haftalık 30 saat T4

### Framework
- Unsloth + TRL >= 0.19 + vLLM
- Base Model: unsloth/Qwen3-8B-bnb-4bit (T4 tek)
- LoRA Rank: 32 (GRPO için)
- QLoRA: 4-bit quantization

### Developer LLM
- Cerebras: qwen-3-32b (1800 tok/sn, en hızlı ücretsiz)
- X.AI (Grok): grok-3 / grok-3-mini (yüksek hızlı)
- Yedek: OpenRouter, Google AI Studio

### Değerlendirme
- LiveBench (MMLU/HumanEval KULLANILMIYOR — veri kirliliği)
- Regresyon kontrolü: önceki skora göre > -2%

### Depolama
- HF Hub: aalpay1968/target-llm-chat (LoRA adapter'lar)
- GitHub: aalpay1968-eng/LLM-DEV (kod + Memory Bank + sonuçlar)
