import os
import json
import time
import subprocess
import sys
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

        print("[CHAT] Sending question to Kaggle...")
        if git_sync_push():
            print("[CHAT] Request sent. Waiting for response (this may take up to 15 mins for initial Kaggle load)...")
            
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
                            print("\nAnaç LLM Yanıtı:")
                            print("-" * 40)
                            print(data.get("answer", "No answer found."))
                            print("-" * 40)
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
