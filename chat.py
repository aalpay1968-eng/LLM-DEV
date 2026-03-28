import os
import json
import time
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path

# Force UTF-8 encoding for Windows terminal output 
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Local project path
PROJECT_DIR = Path(r"E:\000 ALPAY Teknoloji\Teknoloji\LLM-DEV")
REPO_DIR = PROJECT_DIR / "repo"
CHAT_DIR = REPO_DIR / "chat"
REQUEST_FILE = CHAT_DIR / "request.json"
RESPONSE_FILE = CHAT_DIR / "response.json"

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
        question = input("\nUser > ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        # Save request
        CHAT_DIR.mkdir(exist_ok=True)
        with open(REQUEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"question": question, "timestamp": time.time()}, f, indent=2, ensure_ascii=False)

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
