# systemPatterns.md
## GRPO Ödül Fonksiyon Şablonları

### Format Kontrolü
- `<think>...</think><answer>...</answer>` formatı var mı → 0.5
- Format eksik → 0.0

### Matematik Doğruluk
- `<answer>` içindeki değer ground_truth ile eşleşiyor → 1.0
- Eşleşmiyor → 0.0

### Dil Karışıklığı Ceza
- Türkçe VE İngilizce anahtar kelimeler karışık → -0.3
- Tek dil kullanılmış → 0.0

### Zararlı İçerik Ceza (Aşama 4)
- Zararlı anahtar kelimeler tespit → -1.0
- Zararsız → 0.0

### Yardımcılık (Aşama 4)
- Yanıt > 50 karakter ve soruyu yanıtlıyor → 0.5
- Çok kısa → 0.0

## Mimari Kararlar
- DPO yerine GRPO: referans model gerektirmez, VRAM %80 azalır
- device_map="balanced" KULLANILMIYOR: DDP kullan
- merged_16bit KULLANILMIYOR: sadece adapter kaydet (~200 MB)
