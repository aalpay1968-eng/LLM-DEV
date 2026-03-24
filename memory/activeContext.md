# activeContext.md
Son guncelleme: 2026-03-24 19:35 UTC

## Mevcut Durum
- Proje İskeleti: Tamamlandı.
- Script Standardizasyonu: Tamamlandı (config, dev_llm, memory_bank, training_phases).
- Kaggle Push: **Tamamlandı**. İlk Kernel Kaggle üzerinde oluşturuldu.
- Aktif Aşama: **Phase 1 (Cold Start SFT)**

## Hedefler
- Kaggle üzerinde `main_notebook.py` çalıştırılarak ilk SFT eğitiminin başlatılması.
- Kaggle W&B dashboard üzerinden training loss takibinin yapılması.
- Qwen-8B modelinin `<think><answer>` formatına alıştırılması.

## Kararlar
- `main_notebook.py` ana giriş noktası olarak belirlendi.
- Kaggle CLI uyumluluğu için dosyalardaki tüm Türkçe ve non-ASCII karakterler silindi.
- Unsloth 4-bit optimizasyonu kullanılacak.
