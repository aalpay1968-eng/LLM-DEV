# scripts/developer_llm.py — Developer LLM orkestrasyon (ReAct)
import json, time, random, os
from openai import OpenAI


def developer_client_olustur(memory_bank, developer_apis, prompt_fn):
    sistem_prompt = prompt_fn(memory_bank)
    for cfg in developer_apis:
        key = os.environ.get(cfg["key_secret"])
        if not key:
            continue
        try:
            client = OpenAI(api_key=key, base_url=cfg["base_url"])
            test_kwargs = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": sistem_prompt},
                    {"role": "user", "content": "test"},
                ],
                "max_tokens": 5,
            }
            test_kwargs.update(cfg.get("extra_params", {}))
            client.chat.completions.create(**test_kwargs)
            print(f"✅ Developer LLM: {cfg['name']}")
            return client, cfg["model"], cfg.get("extra_params", {}), sistem_prompt
        except Exception as e:
            print(f"⚠️ {cfg['name']}: {e}")
    raise RuntimeError("Tüm Developer LLM API'leri başarısız.")


def developer_llm_cagir(client, model_adi, sistem, kullanici,
                         extra_params=None, maks_deneme=3, langfuse=None,
                         iz_adi="dev_llm"):
    if extra_params is None:
        extra_params = {}
    for deneme in range(maks_deneme):
        trace = None
        if langfuse:
            trace = langfuse.trace(name=iz_adi, input={"kullanici": kullanici[:200]})
        try:
            kwargs = {
                "model": model_adi,
                "messages": [
                    {"role": "system", "content": sistem},
                    {"role": "user", "content": kullanici},
                ],
                "temperature": 0.3,
                "max_tokens": 800,
            }
            kwargs.update(extra_params)
            r = client.chat.completions.create(**kwargs)
            metin = r.choices[0].message.content.strip()
            if trace:
                trace.update(output=metin[:200], metadata={"model": model_adi, "deneme": deneme})
            if "```" in metin:
                metin = metin.split("```")[1]
                if metin.lower().startswith("json"):
                    metin = metin[4:]
            return json.loads(metin.strip())
        except Exception as e:
            if trace:
                trace.update(metadata={"hata": str(e)})
            print(f"⚠️ Developer LLM deneme {deneme+1}/{maks_deneme}: {e}")
            time.sleep(5)
    return {}


def developer_llm_tasarla(client, model_adi, sistem, extra_params,
                           gecmis, asama, langfuse=None):
    sablonu = {
        "deney_id": f"exp_{asama}_{int(time.time())}",
        "asama": asama,
        "thought_process": "ReAct: Önceki sonuçları analiz et...",
        "degisiklik_aciklama": "neden bu parametreler",
        "learning_rate": 5e-6,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "num_generations": 6,
        "max_steps": 100,
        "r_rank": 32,
        "beklenen_gelisme": "hangi metrikte iyileşme",
    }
    kullanici = (
        f"Mevcut eğitim aşaması: {asama}\n"
        f"Son 3 deney: {json.dumps(gecmis[-3:], indent=2, ensure_ascii=False)}\n\n"
        f"ReAct deseni ile düşün, sonra bu şablonu doldur:\n"
        f"{json.dumps(sablonu, indent=2, ensure_ascii=False)}"
    )
    sonuc = developer_llm_cagir(client, model_adi, sistem, kullanici,
                                 extra_params=extra_params, langfuse=langfuse,
                                 iz_adi="tasarla")
    return sonuc if sonuc else {**sablonu, "degisiklik_aciklama": "varsayılan config"}


def developer_llm_yargic(client, model_adi, sistem, extra_params,
                          prompt, yanit_a, yanit_b, langfuse=None):
    if random.random() > 0.5:
        yanit_a, yanit_b = yanit_b, yanit_a
        siralama_degisti = True
    else:
        siralama_degisti = False
    rubrik = {
        "thought_process": "Adım adım analiz...",
        "mantik_tutarli_a": True,
        "mantik_tutarli_b": True,
        "dil_karisikligi_a": False,
        "dil_karisikligi_b": False,
        "tercih": "A",
        "red_sebebi": "varsa açıkla",
    }
    kullanici = (
        f"Soru:\n{prompt}\n\n"
        f"Yanıt A:\n{yanit_a}\n\nYanıt B:\n{yanit_b}\n\n"
        "Değerlendirme kriterleri: mantıksal tutarlılık, dil karışıklığı yok, "
        "döngüsüz akıl yürütme, doğru format.\n"
        "⚠️ Uzun yanıtı kısa yanıttan üstün TUTMA.\n"
        f"Bu şablonu doldur:\n{json.dumps(rubrik, indent=2, ensure_ascii=False)}"
    )
    sonuc = developer_llm_cagir(client, model_adi, sistem, kullanici,
                                 extra_params=extra_params, langfuse=langfuse,
                                 iz_adi="yargic")
    if not sonuc:
        return None
    if siralama_degisti and sonuc.get("tercih") in ("A", "B"):
        sonuc["tercih"] = "B" if sonuc["tercih"] == "A" else "A"
    return sonuc
