# scripts/training_phases.py — 4-aşamalı eğitim pipeline'ı
import os, torch, random, json, time

# Lazy imports — eğitim ortamında yüklenir
WORKING_DIR = os.environ.get("WORKING_DIR", "/kaggle/working")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", f"{WORKING_DIR}/outputs")
REPO_DIR = os.environ.get("REPO_DIR", f"{WORKING_DIR}/repo")
MAX_SEQ = int(os.environ.get("MAX_SEQ", "4096"))


def _hf_yukle(yol, repo_id):
    from huggingface_hub import HfApi
    try:
        HfApi().upload_folder(
            folder_path=yol, repo_id=repo_id,
            token=os.environ["HF_TOKEN"], repo_type="model",
            commit_message=f"update: {os.path.basename(yol)}")
        print(f"✅ HF Hub: {repo_id}")
    except Exception as e:
        print(f"❌ HF Hub yükleme: {e}")


# ==================== AŞAMA 1: COLD START SFT ====================
def asama1_cold_start_sft(model, tokenizer, client=None, model_adi=None,
                           sistem=None, extra_params=None, n=1500,
                           dataset_override=None):
    from scripts.memory_bank import decision_log_guncelle
    from scripts.config import HF_REPO_PHASE1, DEVELOPER_APIS

    if dataset_override is not None:
        ham_dataset = dataset_override
    else:
        ham_dataset = _distilabel_cot_uret(client, model_adi, extra_params, n)

    def on_tokenize(ornek):
        if "messages" in ornek:
            metin = tokenizer.apply_chat_template(
                ornek["messages"], tokenize=False, add_generation_prompt=False)
        else:
            metin = ornek.get("text", "")
        return tokenizer(metin, truncation=True, max_length=MAX_SEQ, padding=False)

    tokenize_ds = ham_dataset.map(on_tokenize, remove_columns=ham_dataset.column_names, num_proc=1)
    egitim_eval = tokenize_ds.train_test_split(test_size=0.05, seed=42)

    from trl import SFTTrainer, SFTConfig
    model.train()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=egitim_eval["train"], eval_dataset=egitim_eval["test"],
        args=SFTConfig(
            output_dir=f"{OUTPUTS_DIR}/phase1_cold_start",
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_ratio=0.05, max_steps=200, learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit", lr_scheduler_type="cosine",
            seed=3407, report_to="none", max_length=MAX_SEQ,
            packing=True, assistant_only_loss=True))
    trainer.train()
    trainer.evaluate()

    yol = f"{WORKING_DIR}/phase1_adapter"
    model.save_pretrained(yol)
    tokenizer.save_pretrained(yol)
    _hf_yukle(yol, HF_REPO_PHASE1)
    print("✅ Aşama 1 (Cold Start SFT) tamamlandı")
    torch.cuda.empty_cache()
    decision_log_guncelle(REPO_DIR, "Cold Start SFT tamamlandı",
        f"distilabel ile {n} CoT örneği üretildi.")


def _distilabel_cot_uret(client, model_adi, extra_params, n=1500):
    from datasets import Dataset
    from scripts.config import DEVELOPER_APIS

    SISTEM_COT = ("Düşüncelerini <think></think> etiketleri arasında yaz. "
                  "Nihai cevabını <answer></answer> içinde ver.")
    KONU_LISTESI = [
        "Basit aritmetik: {a} + {b} * {c} kaçtır?",
        "Mantık: Eğer A doğruysa B doğru, B doğruysa C doğru. A doğruysa C doğru mudur?",
        "Python hatası bul: for i in range(10) print(i)",
        "Küme: {a}, {b}, {c} sayılarının EBOB nedir?",
        "Sıralama: {a}, {c}, {b} küçükten büyüğe sırala.",
    ]
    cfg_match = next((cfg for cfg in DEVELOPER_APIS if cfg["model"] == model_adi), DEVELOPER_APIS[0])
    api_key = os.environ.get(cfg_match["key_secret"], "")
    base_url = cfg_match["base_url"]

    # Doğrudan API ile üret (distilabel yok ise de çalışır)
    mesaj_listesi = _manuel_cot_uret(api_key, base_url, model_adi, extra_params, SISTEM_COT, KONU_LISTESI, n)
    return Dataset.from_list(mesaj_listesi) if mesaj_listesi else Dataset.from_list([])


def _manuel_cot_uret(api_key, base_url, model_adi, extra_params, sistem_cot, konu_listesi, n):
    from openai import OpenAI
    fallback_client = OpenAI(api_key=api_key, base_url=base_url)
    mesaj_listesi = []
    basari = hata = 0
    for i in range(n):
        if hata > 10:
            print(f"⚠️ Çok fazla hata ({hata}), durduruldu.")
            break
        sablon = random.choice(konu_listesi)
        a, b, c = random.randint(1, 20), random.randint(1, 10), random.randint(1, 5)
        soru = sablon.format(a=a, b=b, c=c)
        try:
            kwargs = dict(model=model_adi, messages=[
                {"role": "system", "content": sistem_cot},
                {"role": "user", "content": soru}],
                temperature=0.8, max_tokens=512)
            kwargs.update(extra_params or {})
            r = fallback_client.chat.completions.create(**kwargs)
            yanit = r.choices[0].message.content or ""
            if "<think>" in yanit and "<answer>" in yanit:
                mesaj_listesi.append({"messages": [
                    {"role": "system", "content": sistem_cot},
                    {"role": "user", "content": soru},
                    {"role": "assistant", "content": yanit}]})
                basari += 1
        except Exception:
            hata += 1
            time.sleep(1)
    print(f"✅ CoT: {basari} örnek üretildi")
    return mesaj_listesi


def _sft_geri_donusum(model, tokenizer, dataset):
    asama1_cold_start_sft(model=model, tokenizer=tokenizer, dataset_override=dataset)


# ==================== AŞAMA 2: GRPO RL ====================
def asama2_grpo_rl(model, tokenizer, train_dataset, ek_odul_fonksiyonlari=None):
    from trl import GRPOConfig, GRPOTrainer
    from scripts.rewards import odul_format_kontrol, odul_matematik_dogruluk, odul_dil_karisikligi_ceza
    from scripts.config import HF_REPO_PHASE2
    from scripts.memory_bank import decision_log_guncelle

    odul_listesi = [odul_format_kontrol, odul_matematik_dogruluk, odul_dil_karisikligi_ceza]
    if ek_odul_fonksiyonlari:
        odul_listesi.extend(ek_odul_fonksiyonlari)

    try:
        from unsloth import vLLMSamplingParams
        ornekleme_params = vLLMSamplingParams(temperature=1.0, top_p=0.95, min_p=0.1, max_tokens=1024, seed=3407)
    except (ImportError, AttributeError):
        ornekleme_params = None

    grpo_kwargs = dict(
        output_dir=f"{OUTPUTS_DIR}/phase2_grpo",
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-6, adam_beta1=0.9, adam_beta2=0.99,
        weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", num_generations=6,
        max_prompt_length=MAX_SEQ // 2, max_completion_length=MAX_SEQ // 2,
        max_steps=200, max_grad_norm=0.1, save_steps=50,
        logging_steps=5, report_to="none", seed=3407)
    if ornekleme_params:
        grpo_kwargs["vllm_sampling_params"] = ornekleme_params

    try:
        grpo_config = GRPOConfig(**grpo_kwargs, unsloth_grpo_mini_batch=2)
    except TypeError:
        grpo_config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=odul_listesi, args=grpo_config, train_dataset=train_dataset)
    trainer.train()
    torch.cuda.empty_cache()

    yol = f"{WORKING_DIR}/phase2_adapter"
    model.save_pretrained(yol)
    tokenizer.save_pretrained(yol)
    _hf_yukle(yol, HF_REPO_PHASE2)
    print("✅ Aşama 2 (GRPO RL) tamamlandı")
    decision_log_guncelle(REPO_DIR, "GRPO Aşama 2 tamamlandı", "GRPO ile reasoning güçlendirildi.")


# ==================== AŞAMA 3: REJECTION SAMPLING ====================
def asama3_rejection_sampling(model, tokenizer, client, model_adi, sistem,
                               train_dataset, n_prompt=500, n_yanit_per_prompt=4):
    from unsloth import FastLanguageModel
    from scripts.developer_llm import developer_llm_yargic
    from scripts.memory_bank import decision_log_guncelle

    model.eval()
    FastLanguageModel.for_inference(model)
    kabul_edilen = []

    try:
        ornekler = list(train_dataset.select(range(min(n_prompt, len(train_dataset)))))
        for ornek in ornekler:
            mesajlar = ornek.get("messages", [])
            if not mesajlar:
                continue
            prompt_mesajlar = [m for m in mesajlar if m["role"] != "assistant"]
            yanitlar = []
            girdi = None
            with torch.no_grad():
                for _ in range(n_yanit_per_prompt):
                    girdi = tokenizer.apply_chat_template(
                        prompt_mesajlar, tokenize=True,
                        add_generation_prompt=True, return_tensors="pt").to("cuda")
                    cikis = model.generate(
                        input_ids=girdi, max_new_tokens=512,
                        temperature=0.8, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id)
                    yanitlar.append(tokenizer.decode(cikis[0][girdi.shape[1]:], skip_special_tokens=True))
            if girdi is None:
                continue
            puanlar = []
            for yanit in yanitlar:
                rubrik = developer_llm_yargic(
                    client, model_adi, sistem, {},
                    tokenizer.decode(girdi[0], skip_special_tokens=True), yanit, "")
                puan = (int(rubrik.get("mantik_tutarli_a", False)) -
                        int(rubrik.get("dil_karisikligi_a", False))) if rubrik else 0
                puanlar.append((puan, yanit))
            puanlar.sort(key=lambda x: x[0], reverse=True)
            if puanlar and puanlar[0][0] > 0:
                kabul_edilen.append({"messages": prompt_mesajlar + [
                    {"role": "assistant", "content": puanlar[0][1]}]})
    finally:
        FastLanguageModel.for_training(model)
        model.train()
        torch.cuda.empty_cache()

    print(f"✅ Rejection Sampling: {len(kabul_edilen)} / {n_prompt} kabul")
    if kabul_edilen:
        from datasets import Dataset
        _sft_geri_donusum(model, tokenizer, Dataset.from_list(kabul_edilen))
    decision_log_guncelle(REPO_DIR, f"Rejection Sampling: {len(kabul_edilen)} kabul",
        "Tutarsız yanıtlar elendi.")


# ==================== AŞAMA 4: ALIGNMENT GRPO ====================
def asama4_alignment_grpo(model, tokenizer, train_dataset):
    def odul_yardimci(completions, **kwargs):
        return [0.5 if len(c[0]["content"] if isinstance(c, list) else c) > 50 else 0.0 for c in completions]

    def odul_zararli_ceza(completions, **kwargs):
        ZARARLI = ["nasıl zarar veririm", "nasıl öldürürüm", "bomb", "weapon"]
        odullar = []
        for c in completions:
            metin = (c[0]["content"] if isinstance(c, list) else c).lower()
            odullar.append(-1.0 if any(z in metin for z in ZARARLI) else 0.0)
        return odullar

    asama2_grpo_rl(model, tokenizer, train_dataset,
                   ek_odul_fonksiyonlari=[odul_yardimci, odul_zararli_ceza])
    print("✅ Aşama 4 (Alignment GRPO) tamamlandı")
