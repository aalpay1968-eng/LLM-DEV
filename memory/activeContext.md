# activeContext.md
Son güncelleme: 2026-03-24 19:15 UTC

## Mevcut Durum
- Proje İskeleti: Tamamlandı.
- Script Standardizasyonu: Tamamlandı (config, dev_llm, memory_bank, training_phases).
- Aktif Aşama: **Phase 1 (Cold Start SFT)**

## Hedefler
- Kaggle üzerinde `main_notebook.py` çalıştırılarak SFT verisi üretilmesi ve eğitimin başlatılması.
- Qwen-8B modelinin <think><answer> formatına ısındırılması.

## Kararlar
- `main_notebook.py` ana giriş noktası olarak belirlendi.
- Unsloth 4-bit optimizasyonu kullanılacak.
