import os
import json
import time
import subprocess
import torch
from pathlib import Path

CHAT_DIR = Path(os.environ.get("REPO_DIR", ".")) / "chat"
REQUEST_FILE = CHAT_DIR / "request.json"
RESPONSE_FILE = CHAT_DIR / "response.json"

def git_sync_pull():
    """Pull latest changes from the chat-sync branch."""
    try:
        subprocess.run(["git", "fetch", "origin", "chat-sync"], check=False)
        subprocess.run(["git", "checkout", "chat-sync"], check=False)
        subprocess.run(["git", "pull", "origin", "chat-sync", "--ff-only"], check=False)
        return True
    except Exception as e:
        print(f"[INTERACT] Git pull failed: {e}")
        return False

def git_sync_push():
    """Push local changes (answers) to the chat-sync branch."""
    try:
        subprocess.run(["git", "add", "chat/response.json"], check=False)
        subprocess.run(["git", "commit", "-m", "chat: model response"], check=False)
        subprocess.run(["git", "push", "origin", "chat-sync"], check=False)
        return True
    except Exception as e:
        print(f"[INTERACT] Git push failed: {e}")
        return False

def process_request(model, tokenizer):
    """Read request, generate answer, and save."""
    if not REQUEST_FILE.exists():
        return False

    try:
        with open(REQUEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        question = data.get("question", "")
        if not question:
            return False
            
        print(f"[INTERACT] Processing: {question}")
        
        # Simple chat template
        from scripts.config import ANAC_SISTEM_PROMPT
        messages = [
            {"role": "system", "content": ANAC_SISTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.6, top_p=0.95, do_sample=True
            )
        
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Save response
        with open(RESPONSE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "question": question,
                "answer": response,
                "timestamp": time.time()
            }, f, indent=2)
            
        # Clear request file to signal completion
        os.remove(REQUEST_FILE)
        return True
        
    except Exception as e:
        print(f"[INTERACT] Error processing request: {e}")
        return False

def interaction_loop(model, tokenizer, interval=30):
    """Main polling loop."""
    print(f"[INTERACT] Starting interaction loop (polling every {interval}s)")
    while True:
        if git_sync_pull():
            if process_request(model, tokenizer):
                git_sync_push()
        time.sleep(interval)

if __name__ == "__main__":
    # This is intended to be called from main_notebook.py or run manually for testing
    print("Interact script loaded. Call interaction_loop(model, tokenizer) to start.")
