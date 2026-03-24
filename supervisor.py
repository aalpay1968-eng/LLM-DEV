#!/usr/bin/env python3
"""
LLM-DEV Supervisor -- 7/24 Continuous Training
===============================================
Monitors Kaggle kernel status and auto-restarts when complete.
Also serves a local web dashboard for monitoring + chat access.

Usage:
    python supervisor.py

Opens http://localhost:8585 in browser automatically.
"""

import os
import sys
import json
import time
import subprocess
import threading
import http.server
import socketserver
import webbrowser
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(r"E:\000 ALPAY Teknoloji\Teknoloji\LLM-DEV")
KERNEL_SLUG = "adilalpay/llm-dev-v1"
DASHBOARD_PORT = 8585
CHECK_INTERVAL = 120  # seconds between status checks
MAX_RESTARTS = 50     # max auto-restarts (safety)

# State file for tracking
STATE_FILE = PROJECT_DIR / "supervisor_state.json"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "total_runs": 0,
        "current_status": "unknown",
        "last_check": None,
        "gradio_url": None,
        "phase": "phase2",
        "history": [],
    }


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8")


def kaggle_status():
    """Check Kaggle kernel status via CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace")
        output = result.stdout.strip()
        if "complete" in output.lower():
            return "complete"
        elif "running" in output.lower():
            return "running"
        elif "error" in output.lower():
            return "error"
        elif "queued" in output.lower():
            return "queued"
        return output.lower() if output else "unknown"
    except Exception as e:
        return f"check_failed: {e}"


def kaggle_output():
    """Fetch kernel output to extract Gradio URL."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", KERNEL_SLUG,
             "-p", str(PROJECT_DIR / "kernel_output")],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
        # Try to find Gradio share URL in output
        lines = result.stdout + result.stderr
        for line in lines.split("\n"):
            if "gradio.live" in line or "share" in line.lower():
                import re
                match = re.search(r"https?://\S+\.gradio\.live\S*", line)
                if match:
                    return match.group(0)
        # Also check log file
        log_dir = PROJECT_DIR / "kernel_output"
        if log_dir.exists():
            for f in log_dir.glob("*.log"):
                content = f.read_text(encoding="utf-8", errors="replace")
                import re
                match = re.search(
                    r"https?://\S+\.gradio\.live\S*", content)
                if match:
                    return match.group(0)
    except Exception:
        pass
    return None


def kaggle_push():
    """Push a new kernel version to restart training."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(PROJECT_DIR)],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
        return "successfully pushed" in result.stdout.lower()
    except Exception:
        return False


def supervisor_loop(state):
    """Main supervisor loop -- monitors and auto-restarts."""
    print(f"[SUPERVISOR] Started. Checking every {CHECK_INTERVAL}s")
    print(f"[SUPERVISOR] Kernel: {KERNEL_SLUG}")
    print(f"[SUPERVISOR] Dashboard: http://localhost:{DASHBOARD_PORT}")

    while state["total_runs"] < MAX_RESTARTS:
        try:
            status = kaggle_status()
            state["current_status"] = status
            state["last_check"] = datetime.now().isoformat()

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Status: {status} | "
                  f"Runs: {state['total_runs']}")

            if status == "running":
                # Try to get Gradio URL
                url = kaggle_output()
                if url:
                    state["gradio_url"] = url
                    print(f"[{ts}] Gradio: {url}")

            elif status == "complete":
                state["history"].append({
                    "run": state["total_runs"],
                    "completed_at": datetime.now().isoformat(),
                    "phase": state["phase"],
                })
                print(f"[{ts}] Kernel completed. Auto-restarting...")
                time.sleep(10)

                if kaggle_push():
                    state["total_runs"] += 1
                    state["current_status"] = "restarting"
                    print(f"[{ts}] Kernel v{state['total_runs']+2} "
                          f"pushed successfully!")
                else:
                    state["current_status"] = "push_failed"
                    print(f"[{ts}] Push failed! Retrying in 5 min...")
                    time.sleep(300)
                    continue

            elif status == "error":
                state["history"].append({
                    "run": state["total_runs"],
                    "error_at": datetime.now().isoformat(),
                    "phase": state["phase"],
                })
                print(f"[{ts}] Kernel error! Waiting 5 min then retry...")
                time.sleep(300)
                if kaggle_push():
                    state["total_runs"] += 1
                    print(f"[{ts}] Recovery push successful.")

            save_state(state)
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n[SUPERVISOR] Stopped by user.")
            save_state(state)
            break
        except Exception as e:
            print(f"[SUPERVISOR] Error: {e}")
            time.sleep(60)


# ============================================================
# Dashboard HTTP Server
# ============================================================
class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            state = load_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps(state, default=str).encode("utf-8"))
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        pass  # Suppress HTTP logs


def start_dashboard():
    """Start the local dashboard server."""
    DASHBOARD_DIR.mkdir(exist_ok=True)
    with socketserver.TCPServer(
            ("", DASHBOARD_PORT), DashboardHandler) as httpd:
        httpd.serve_forever()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    state = load_state()

    # Start dashboard in background
    dash_thread = threading.Thread(target=start_dashboard, daemon=True)
    dash_thread.start()
    print(f"[DASHBOARD] http://localhost:{DASHBOARD_PORT}")

    # Open browser
    time.sleep(1)
    webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")

    # Start supervisor loop
    supervisor_loop(state)
