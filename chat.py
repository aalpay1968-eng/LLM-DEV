import os
import json
import time
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# Force UTF-8 encoding for Windows terminal output 
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Local project path
PROJECT_DIR = Path(r"E:\000 ALPAY Teknoloji\Teknoloji\LLM-DEV")
REPO_DIR = PROJECT_DIR / "repo"
CHAT_DIR = REPO_DIR / "chat"
REQUEST_FILE = CHAT_DIR / "request.json"
RESPONSE_FILE = CHAT_DIR / "response.json"
HISTORY_FILE = CHAT_DIR / "history.json"

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_context_string(history):
    if not history:
        return ""
    # Keep last 3 interactions to avoid token limits
    recent = history[-3:]
    context = "GEÇMİŞ KONUŞMALARIMIZ:\n"
    for item in recent:
        context += f"Kullanıcı: {item['user']}\nSen (Anaç LLM): {item['bot']}\n\n"
    context += "YENİ SORU:\n"
    return context

def speak_text(text):
    print("\n[Sesli Okuma Başlıyor...]")
    # Remove some markdown special chars for better TTS reading
    clean_text = re.sub(r'[*_#`]', '', text)
    tts_cmd = [
        "edge-tts",
        "--voice", "tr-TR-EmelNeural",
        "--text", clean_text,
        "--write-media", "temp_speech.mp3"
    ]
    subprocess.run(tts_cmd, capture_output=True)
    if os.path.exists("temp_speech.mp3"):
        if os.name == 'nt':
            os.startfile("temp_speech.mp3")
        else:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "temp_speech.mp3"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def listen_to_mic():
    if sr is None:
        print("[CHAT] speech_recognition kütüphanesi yüklü değil. Sesli giriş kullanılamaz.")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Mikrofon Dinleniyor... Lütfen konuşun (Sessizlikte kapanır)]")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            print("[CHAT] Ses algılandı, metne çevriliyor...")
            text = r.recognize_google(audio, language="tr-TR")
            return text
        except sr.WaitTimeoutError:
            print("[CHAT] Ses algılanmadı.")
            return ""
        except sr.UnknownValueError:
            print("[CHAT] Sesi anlayamadım.")
            return ""
        except Exception as e:
            print(f"[CHAT] Mikrofon hatası: {e}")
            return ""

def git_sync_push():
    """Push local question to the chat-sync branch."""
    try:
        subprocess.run(["git", "-C", str(REPO_DIR), "add", "chat/request.json"], check=False, capture_output=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "commit", "-m", f"chat: user request {time.time()}"], check=False, capture_output=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "push", "origin", "chat-sync"], check=False, capture_output=True)
        return True
    except Exception as e:
        print(f"[CHAT] Push failed: {e}")
        return False

def git_sync_pull():
    """Pull model interaction from the chat-sync branch."""
    try:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "chat-sync"], check=False, capture_output=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "origin", "chat-sync", "--ff-only"], check=False, capture_output=True)
        return True
    except Exception as e:
        print(f"[CHAT] Pull failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Anaç (Maternal) LLM: In-Kaggle Chat CLI (v1.1)")
    print("Bir sorununuz veya danışmak istediğiniz bir konu var mı?")
    print("Type 'exit' to quit.")
    print("-" * 60)

    # Pre-check branch
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "chat-sync"], check=False, capture_output=True)

    while True:
        question = input("\nSen (Yazın veya sesli sormak için boş bırakıp ENTER'a basın) > ").strip()
        if not question:
            question = listen_to_mic()
            if not question:
                continue
            print(f"Kullanıcı (Sesli): {question}")

        if question.lower() in ("exit", "quit", "q"):
            break

        history = load_history()
        context_str = get_context_string(history)
        full_payload = context_str + question if context_str else question

        # Save request
        CHAT_DIR.mkdir(exist_ok=True)
        with open(REQUEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"question": full_payload, "timestamp": time.time()}, f, indent=2, ensure_ascii=False)

        project_start = datetime(2026, 3, 27, 20, 0, 0)
        hours_developed = max(1, int((datetime.now() - project_start).total_seconds() / 3600))

        print(f"\n[Anaç LLM] Zeka Seviyesi: Gelişmekte (Çocuk Öğrenme Aşaması)")
        print(f"[Anaç LLM] {hours_developed} saattir büyüyorum ve yeni şeyler öğreniyorum.")
        print("[Anaç LLM] Sorunu analiz edip düşünüyorum...\n")

        print("[CHAT] Soru Kaggle üzerindeki modele iletiliyor...")
        if git_sync_push():
            print("[CHAT] Gönderildi. Modelin düşünmesi ve cevap vermesi bekleniyor (Yaklaşık 1-15 dk)...")
            
            # Polling for response
            start_time = time.time()
            found = False
            while time.time() - start_time < 900:  # 15 min timeout
                time.sleep(20)
                git_sync_pull()
                if RESPONSE_FILE.exists():
                    try:
                        with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("question") == question:
                            raw_answer = data.get("answer", "Yanıt bulunamadı.")
                            print("\n\n" + "=" * 60)
                            
                            think_match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
                            if think_match:
                                think_text = think_match.group(1).strip()
                                print("--- (İçsel Düşüncelerim / Merakım) ---")
                                print(think_text)
                                print("--------------------------------------\n")
                                raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                                
                            answer_match = re.search(r"<answer>(.*?)</answer>", raw_answer, re.DOTALL)
                            if answer_match:
                                final_answer = answer_match.group(1).strip()
                            else:
                                final_answer = raw_answer.strip()
                                
                            print("Sana Şefkatli Yanıtım:")
                            print("-" * 40)
                            print(final_answer)
                            print("=" * 60 + "\n")
                            
                            # Tarihçeyi Kaydet
                            history.append({"user": question, "bot": final_answer})
                            save_history(history)
                            
                            # Sesli Oku
                            speak_text(final_answer)
                            
                            found = True
                            # Clean up
                            os.remove(RESPONSE_FILE)
                            break
                    except Exception:
                        pass
                print(".", end="", flush=True)
            
            if not found:
                print("\n[CHAT] Timeout waiting for response. Check Kaggle kernel status.")
        else:
            print("[CHAT] Failed to send request.")

if __name__ == "__main__":
    main()
