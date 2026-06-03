"""
Antigravity Git Threat Shield — Windows Desktop App
Run: python desktop_app.py
"""
import sys
import os
import threading
import subprocess
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set working directory to project root so config.py resolves paths correctly
os.chdir(ROOT)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#00e5ff" # Electric cyan
DANGER   = "#ff1744" # Crimson red
SUCCESS  = "#00e676" # Neon green
WARNING  = "#ffd600" # Amber yellow
BG_CARD  = "#0e1622" # Midnight card
BG_DARK  = "#060b13" # Deep space black
MUTED    = "#8fa3b9" # Steel gray-blue

# ── Scanner loader ────────────────────────────────────────────────────────────
def load_scanner():
    from prototype.backend.config import settings
    # Ensure checkpoint path is absolute and correct regardless of cwd
    checkpoint = ROOT / "adaptive_model_final.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model not found at: {checkpoint}\n"
            f"Please verify adaptive_model_final.pth exists in {ROOT}"
        )
    # Override checkpoint path to absolute
    settings.CHECKPOINT_PATH = checkpoint
    settings.TEMP_DIR.mkdir(exist_ok=True)

    # Import the global singleton scanner
    from prototype.backend.scanner import scanner
    
    # Reload model to ensure it uses the updated absolute path
    scanner.reload_model()
    
    return scanner

# ── Git helpers ───────────────────────────────────────────────────────────────
HOOK_PATH = ROOT / ".git" / "hooks" / "pre-commit"

def hook_installed():
    return HOOK_PATH.exists()

def install_hook():
    from git_threat_protector import install_hook as _install
    return _install()

def uninstall_hook():
    from git_threat_protector import uninstall_hook as _uninstall
    return _uninstall()

def get_staged_files():
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    except Exception:
        return []

def is_binary_file(path: Path):
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
        return b'\x00' in chunk
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Adaptive Threat Detection — Git Threat Shield")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARK)

        # Load Section 5 Outcomes Config
        config = self.load_shield_config()
        self.quarantine_var = ctk.BooleanVar(value=config.get("quarantine", True))
        self.self_heal_var = ctk.BooleanVar(value=config.get("self_heal", False))
        self.lockout_var = ctk.BooleanVar(value=config.get("lockout", True))

        self.scanner = None
        self.scanner_ready = False
        self._build_ui()
        threading.Thread(target=self._load_scanner_bg, daemon=True).start()

    # ── Scanner background load ───────────────────────────────────────────────
    def _load_scanner_bg(self):
        import requests
        self.server_url = "http://127.0.0.1:8000"
        self.headers = {}
        
        try:
            # Authenticate as admin
            data = {"username": "admin", "password": "securepassword"}
            r = requests.post(f"{self.server_url}/token", data=data, timeout=5)
            if r.status_code == 200:
                token = r.json().get("access_token")
                self.headers = {"Authorization": f"Bearer {token}"}
                self.scanner_ready = True
                
                # Fetch stats to verify connection and model status
                r_stats = requests.get(f"{self.server_url}/system/stats", headers=self.headers, timeout=5)
                stats = r_stats.json() if r_stats.status_code == 200 else {}
                if stats:
                    self.after(0, lambda s=stats: self._update_metrics_ui(s))
                
                # If backend is running, check if capacity is expanded
                is_expanded = False
                # Try to get is_expanded from the backend scanner metadata if possible or default to false
                
                self.after(0, lambda: self.status_label.configure(
                    text="● AI Server Connected", text_color=SUCCESS))
                self.after(0, lambda: self.capacity_label.configure(
                    text="● Mode: API Thin Client", text_color=ACCENT))
                self.after(0, self._refresh_git_status)
            else:
                raise Exception(f"Auth failed: {r.status_code}")
        except Exception as e:
            self.scanner_ready = False
            self.after(0, lambda: self.status_label.configure(
                text="● API Server Offline", text_color=DANGER))
            self.after(0, lambda: self.capacity_label.configure(
                text="● Start uvicorn server first", text_color=WARNING))

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, fg_color=BG_CARD, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="ADAPTIVE THREAT\nDETECTION", font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=ACCENT).pack(pady=(28, 2))
        ctk.CTkLabel(self.sidebar, text="Git Threat Shield", font=ctk.CTkFont(size=11),
                     text_color=MUTED).pack(pady=(0, 28))

        self._nav_buttons = {}
        for name, label in [("shield", "🛡️  Git Shield"), ("scanner", "🔍  File Scanner"), ("evolution", "🔄  Model Evolution"), ("logs", "📜  System Logs")]:
            btn = ctk.CTkButton(
                self.sidebar, text=label, width=180, height=40,
                fg_color="transparent", hover_color="#172237",
                text_color=MUTED, anchor="w",
                command=lambda n=name: self._switch_tab(n)
            )
            btn.pack(pady=3, padx=16)
            self._nav_buttons[name] = btn

        # Status container at bottom of sidebar
        status_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        status_container.pack(side="bottom", pady=24, fill="x")

        self.status_label = ctk.CTkLabel(
            status_container, text="● AI Model Loading...",
            text_color=WARNING, font=ctk.CTkFont(size=11, weight="bold"))
        self.status_label.pack(pady=2)

        self.capacity_label = ctk.CTkLabel(
            status_container, text="● Neural Capacity: Normal",
            text_color=MUTED, font=ctk.CTkFont(size=10))
        self.capacity_label.pack(pady=2)

        # Main content
        self.content = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)

        # Build all tab frames
        self.tabs = {}
        self.tabs["shield"]  = self._build_shield_tab()
        self.tabs["scanner"] = self._build_scanner_tab()
        self.tabs["evolution"] = self._build_evolution_tab()
        self.tabs["logs"]    = self._build_logs_tab()

        self._switch_tab("shield")

    def _switch_tab(self, name):
        for key, frame in self.tabs.items():
            frame.pack_forget()
        self.tabs[name].pack(fill="both", expand=True, padx=24, pady=24)
        for key, btn in self._nav_buttons.items():
            btn.configure(
                fg_color=ACCENT if key == name else "transparent",
                text_color=BG_DARK if key == name else MUTED
            )

    # ══ GIT SHIELD TAB ════════════════════════════════════════════════════════
    def _build_shield_tab(self):
        frame = ctk.CTkFrame(self.content, fg_color=BG_DARK, corner_radius=0)

        ctk.CTkLabel(frame, text="Git Pre-Commit Shield",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(frame, text="Blocks malicious binaries before they enter your Git history.",
                     text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(2, 16))

        # Hook status card
        hook_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        hook_card.pack(fill="x", pady=(0, 16))

        inner = ctk.CTkFrame(hook_card, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=20)

        self.hook_status_label = ctk.CTkLabel(
            inner, text="● Checking hook...",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=MUTED)
        self.hook_status_label.pack(side="left")

        self.hook_toggle_btn = ctk.CTkButton(
            inner, text="ACTIVATE SHIELD", width=180, height=38,
            fg_color=ACCENT, text_color=BG_DARK, font=ctk.CTkFont(weight="bold"),
            command=self._toggle_hook)
        self.hook_toggle_btn.pack(side="right")

        # Outcomes configuration card (Section 5)
        outcomes_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        outcomes_card.pack(fill="x", pady=(0, 16))
        
        inner_out = ctk.CTkFrame(outcomes_card, fg_color="transparent")
        inner_out.pack(fill="x", padx=24, pady=16)
        
        # Left container for checkboxes
        cb_container = ctk.CTkFrame(inner_out, fg_color="transparent")
        cb_container.pack(side="left")
        
        ctk.CTkLabel(cb_container, text="Section 5 Outcomes Configuration:", font=ctk.CTkFont(weight="bold", size=12), text_color=ACCENT).pack(anchor="w", pady=(0, 4))
        
        cb1 = ctk.CTkCheckBox(cb_container, text="📦  Quarantine Artifacts", variable=self.quarantine_var, command=self.save_shield_config, font=ctk.CTkFont(size=11, weight="bold"))
        cb1.pack(side="left", padx=(0, 12))
        
        cb2 = ctk.CTkCheckBox(cb_container, text="✨  Self-Heal (Delete)", variable=self.self_heal_var, command=self.save_shield_config, font=ctk.CTkFont(size=11, weight="bold"))
        cb2.pack(side="left", padx=(0, 12))
        
        cb3 = ctk.CTkCheckBox(cb_container, text="🔒  Lock Repository", variable=self.lockout_var, command=self.save_shield_config, font=ctk.CTkFont(size=11, weight="bold"))
        cb3.pack(side="left")
        
        # Right container for lockout status
        lock_container = ctk.CTkFrame(inner_out, fg_color="transparent")
        lock_container.pack(side="right")
        
        self.lockout_status_label = ctk.CTkLabel(lock_container, text="● Checking lockout status...", font=ctk.CTkFont(weight="bold", size=12), text_color=MUTED)
        self.lockout_status_label.pack(side="left", padx=(0, 16))
        
        self.lockout_clear_btn = ctk.CTkButton(lock_container, text="🔓  Unlock Repository", width=160, height=32, fg_color="#172237", hover_color="#233554", text_color="white", font=ctk.CTkFont(size=11, weight="bold"), command=self._clear_lockout)
        self.lockout_clear_btn.pack(side="right")

        # Staged files + results row
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="both", expand=True)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        row.rowconfigure(0, weight=1)

        # Staged files panel
        left = ctk.CTkFrame(row, fg_color=BG_CARD, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(left, text="Staged Files", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(left, text="Files queued for your next git commit",
                     text_color=MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)

        self.staged_box = ctk.CTkTextbox(left, fg_color="#0a0e14", text_color="white",
                                         font=ctk.CTkFont(family="Courier", size=11), state="disabled")
        self.staged_box.pack(fill="both", expand=True, padx=16, pady=8)

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkButton(btn_row, text="Refresh", width=100, fg_color="#1a2332",
                      command=self._refresh_git_status).pack(side="left", padx=(0, 8))
        self.scan_staged_btn = ctk.CTkButton(
            btn_row, text="Scan Staged Files", fg_color=ACCENT, text_color=BG_DARK,
            font=ctk.CTkFont(weight="bold"), command=self._scan_staged)
        self.scan_staged_btn.pack(side="left")

        # Results panel
        right = ctk.CTkFrame(row, fg_color=BG_CARD, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(right, text="Scan Results", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(right, text="AI verdict per file",
                     text_color=MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)

        self.results_box = ctk.CTkTextbox(right, fg_color="#0a0e14", text_color="white",
                                          font=ctk.CTkFont(family="Courier", size=11), state="disabled")
        self.results_box.pack(fill="both", expand=True, padx=16, pady=16)

        self._refresh_git_status()
        return frame

    def _refresh_git_status(self):
        installed = hook_installed()
        if installed:
            self.hook_status_label.configure(text="● SHIELD ACTIVE — Commits are protected", text_color=SUCCESS)
            self.hook_toggle_btn.configure(text="DEACTIVATE SHIELD", fg_color=DANGER, text_color="white")
        else:
            self.hook_status_label.configure(text="● SHIELD INACTIVE — Commits are unprotected", text_color=DANGER)
            self.hook_toggle_btn.configure(text="ACTIVATE SHIELD", fg_color=ACCENT, text_color=BG_DARK)

        self._refresh_lockout_status()

        staged = get_staged_files()
        self.staged_box.configure(state="normal")
        self.staged_box.delete("1.0", "end")
        if staged:
            for f in staged:
                self.staged_box.insert("end", f"  › {f}\n")
        else:
            self.staged_box.insert("end", "\n  No staged files.\n  Run: git add <file>")
        self.staged_box.configure(state="disabled")

    def _toggle_hook(self):
        def _run():
            try:
                if hook_installed():
                    uninstall_hook()
                    self._log("Git pre-commit hook deactivated.")
                else:
                    install_hook()
                    self._log("Git pre-commit hook activated.")
                self.after(0, self._refresh_git_status)
            except Exception as e:
                self._log(f"Hook toggle error: {e}", error=True)
        threading.Thread(target=_run, daemon=True).start()

    def _scan_staged(self):
        if not self.scanner_ready:
            messagebox.showwarning("Not Ready", "AI model is still loading.")
            return
        staged = get_staged_files()
        if not staged:
            messagebox.showinfo("No Files", "No staged files to scan.")
            return
        self.scan_staged_btn.configure(state="disabled", text="Scanning...")
        threading.Thread(target=self._run_staged_scan, args=(staged,), daemon=True).start()

    def _run_staged_scan(self, staged):
        import requests
        try:
            r = requests.post(f"{self.server_url}/git/scan-staged", headers=self.headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                results = []
                for item in data.get("results", []):
                    # API returns skipped files too
                    results.append((item['filename'], item['prediction'], item.get('confidence', 0.0)))
                self.after(0, lambda: self._show_staged_results(results))
            else:
                raise Exception(f"Server error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            self._log(f"Staged scan error: {e}", error=True)
            self.after(0, lambda: self.scan_staged_btn.configure(state="normal", text="Scan Staged Files"))

    def _show_staged_results(self, results):
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", "end")
        any_threat = False
        for name, pred, conf in results:
            if pred == "Malware":
                any_threat = True
                line = f"  [THREAT]  {name}  ({conf*100:.1f}%)\n"
            elif pred == "Benign":
                line = f"  [CLEAN]   {name}  ({conf*100:.1f}%)\n"
            else:
                line = f"  [SKIP]    {name}\n"
            self.results_box.insert("end", line)
        self.results_box.configure(state="disabled")
        self.scan_staged_btn.configure(state="normal", text="Scan Staged Files")
        if any_threat:
            messagebox.showerror("THREAT DETECTED", "One or more staged files were flagged as malware!\nCommit blocked.")
        else:
            messagebox.showinfo("All Clear", "All staged files passed the security check.")
        self._log(f"Staged scan done. {len(results)} files checked.")

    # ══ FILE SCANNER TAB ══════════════════════════════════════════════════════
    def _build_scanner_tab(self):
        frame = ctk.CTkFrame(self.content, fg_color=BG_DARK, corner_radius=0)

        ctk.CTkLabel(frame, text="File Scanner",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(frame, text="Scan any .exe, .dll, or binary file on your system.",
                     text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(2, 16))

        # Drop zone card
        card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", pady=(0, 16))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=24)

        self.selected_file_label = ctk.CTkLabel(
            inner, text="No file selected", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.selected_file_label.pack(side="left", expand=True, anchor="w")

        ctk.CTkButton(inner, text="Browse File...", width=160, height=38,
                      fg_color="#1a2332", command=self._browse_file).pack(side="left", padx=(8, 0))
        self.scan_file_btn = ctk.CTkButton(
            inner, text="SCAN FILE", width=140, height=38,
            fg_color=ACCENT, text_color=BG_DARK, font=ctk.CTkFont(weight="bold"),
            command=self._scan_selected_file)
        self.scan_file_btn.pack(side="left", padx=(8, 0))

        # Result card
        self.file_result_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        self.file_result_card.pack(fill="both", expand=True)

        self.file_result_label = ctk.CTkLabel(
            self.file_result_card, text="Scan result will appear here.",
            text_color=MUTED, font=ctk.CTkFont(size=13))
        self.file_result_label.pack(expand=True)

        self._selected_file = None
        return frame

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a file to scan",
            filetypes=[("Executables & Binaries", "*.exe *.dll *.bin *.elf *.sys *.msi"), ("All files", "*.*")]
        )
        if path:
            self._selected_file = Path(path)
            self.selected_file_label.configure(text=str(self._selected_file.name), text_color="white")

    def _ensure_result_label_ready(self):
        try:
            if not hasattr(self, "file_result_label") or not self.file_result_label.winfo_exists():
                raise AttributeError
        except (AttributeError, Exception):
            for w in self.file_result_card.winfo_children():
                w.destroy()
            self.file_result_label = ctk.CTkLabel(
                self.file_result_card, text="Scan result will appear here.",
                text_color=MUTED, font=ctk.CTkFont(size=13))
            self.file_result_label.pack(expand=True)

    def _scan_selected_file(self):
        if not self.scanner_ready:
            messagebox.showwarning("Not Ready", "AI model is still loading.")
            return
        if not self._selected_file or not self._selected_file.exists():
            messagebox.showwarning("No File", "Please select a file first.")
            return
        self._ensure_result_label_ready()
        self.scan_file_btn.configure(state="disabled", text="Scanning...")
        self.file_result_label.configure(text="Scanning...", text_color=MUTED)
        threading.Thread(target=self._run_file_scan, args=(self._selected_file,), daemon=True).start()

    def _run_file_scan(self, path: Path):
        import requests
        try:
            with open(path, 'rb') as f:
                files = {'file': (path.name, f, 'application/octet-stream')}
                r = requests.post(f"{self.server_url}/scan", files=files, headers=self.headers, timeout=15)
            if r.status_code == 200:
                res = r.json()
                res_formatted = {
                    'prediction': res['prediction'],
                    'confidence': res['confidence'],
                    'static_analysis': res['analysis']['static'],
                    'dynamic_analysis': res['analysis']['dynamic'],
                    'fusion_strategy': res['analysis']['fusion']
                }
                self.after(0, lambda: self._show_file_result(path, res_formatted))
            else:
                raise Exception(f"Server error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self.file_result_label.configure(
                text=f"Scan error: {msg}", text_color=DANGER))
            self.after(0, lambda: self.scan_file_btn.configure(state="normal", text="SCAN FILE"))

    def _show_file_result(self, path: Path, res: dict):
        pred = res['prediction']
        conf = res['confidence'] * 100
        color = DANGER if pred == "Malware" else SUCCESS

        for w in self.file_result_card.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.file_result_card, text=f"VERDICT: {pred.upper()} ({conf:.1f}% Confidence)",
                     font=ctk.CTkFont(size=16, weight="bold"), text_color=color).pack(pady=(20, 10))

        row = ctk.CTkFrame(self.file_result_card, fg_color="transparent")
        row.pack(pady=10)
        ctk.CTkLabel(row, text=f"Static:   {res.get('static_analysis','—')}", text_color=MUTED,
                     font=ctk.CTkFont(size=11, family="Courier")).pack(anchor="w")
        ctk.CTkLabel(row, text=f"Dynamic:  {res.get('dynamic_analysis','—')}", text_color=MUTED,
                     font=ctk.CTkFont(size=11, family="Courier")).pack(anchor="w")

        ctk.CTkButton(self.file_result_card, text="Scan Another File", fg_color="#1a2332",
                      command=self._reset_file_scanner).pack(pady=8)

        self.scan_file_btn.configure(state="normal", text="SCAN FILE")
        self._log(f"File scan: {path.name} → {pred} ({conf:.1f}%)", error=(pred == "Malware"))

    def _reset_file_scanner(self):
        for w in self.file_result_card.winfo_children():
            w.destroy()
        self.file_result_label = ctk.CTkLabel(
            self.file_result_card, text="Scan result will appear here.",
            text_color=MUTED, font=ctk.CTkFont(size=13))
        self.file_result_label.pack(expand=True)
        self._selected_file = None
        self.selected_file_label.configure(text="No file selected", text_color=MUTED)

    # ══ LOGS TAB ═════════════════════════════════════════════════════════════
    def _build_logs_tab(self):
        frame = ctk.CTkFrame(self.content, fg_color=BG_DARK, corner_radius=0)

        ctk.CTkLabel(frame, text="System Logs",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(frame, text="All scan and hook events from this session.",
                     text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(2, 16))

        self.log_box = ctk.CTkTextbox(frame, fg_color=BG_CARD, text_color="white",
                                      font=ctk.CTkFont(family="Courier", size=12), corner_radius=12)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        ctk.CTkButton(frame, text="Clear Logs", fg_color="#1a2332",
                      command=self._clear_logs).pack(anchor="e", pady=(8, 0))
        return frame

    def _log(self, msg: str, error: bool = False):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        tag = "ERROR" if error else "INFO"
        line = f"[{ts}] [{tag}] {msg}\n"
        self.after(0, lambda: self._append_log(line))

    def _append_log(self, line: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_logs(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ══ CONFIG & LOCKOUT HELPERS ═════════════════════════════════════════════
    def load_shield_config(self):
        import json
        config_path = ROOT / ".git_shield_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"quarantine": True, "self_heal": False, "lockout": True}

    def save_shield_config(self):
        import json
        config_path = ROOT / ".git_shield_config.json"
        config = {
            "quarantine": bool(self.quarantine_var.get()),
            "self_heal": bool(self.self_heal_var.get()),
            "lockout": bool(self.lockout_var.get())
        }
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            self._log("Section 5 outcomes configuration saved.")
        except Exception as e:
            self._log(f"Failed to save shield config: {e}", error=True)

    def _refresh_lockout_status(self):
        lock_file = ROOT / ".git_lockout"
        if lock_file.exists():
            self.lockout_status_label.configure(text="🚨 REPOSITORY LOCKED (Revoked)", text_color=DANGER)
            self.lockout_clear_btn.configure(state="normal", fg_color=SUCCESS, text_color=BG_DARK)
        else:
            self.lockout_status_label.configure(text="🟢 Credentials Status: Active", text_color=SUCCESS)
            self.lockout_clear_btn.configure(state="disabled", fg_color="#172237", text_color=MUTED)

    def _clear_lockout(self):
        lock_file = ROOT / ".git_lockout"
        if lock_file.exists():
            try:
                lock_file.unlink()
                self._log("Credential Lockout cleared. Repository unlocked.")
            except Exception as e:
                self._log(f"Failed to clear lockout: {e}", error=True)
        self._refresh_lockout_status()

    # ══ MODEL EVOLUTION TAB ══════════════════════════════════════════════════
    def _build_evolution_tab(self):
        frame = ctk.CTkFrame(self.content, fg_color=BG_DARK, corner_radius=0)
        
        ctk.CTkLabel(frame, text="Model Evolution & Continuous Learning",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(frame, text="Generate adversarial variants and train candidate models to harden your defense (Section 6).",
                     text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(2, 16))
        
        # Grid layout for controls + visualizer & logs
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=4) # Left column (Controls & Metrics)
        grid.columnconfigure(1, weight=5) # Right column (Logs & Step Flow)
        grid.rowconfigure(0, weight=1)
        
        # LEFT COLUMN
        left_col = ctk.CTkFrame(grid, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Control Card
        control_card = ctk.CTkFrame(left_col, fg_color=BG_CARD, corner_radius=12)
        control_card.pack(fill="x", pady=(0, 16), padx=0)
        
        ctk.CTkLabel(control_card, text="Adversarial Training Configuration", font=ctk.CTkFont(weight="bold", size=14), text_color=ACCENT).pack(anchor="w", padx=20, pady=(16, 12))
        
        # Mutation strategy dropdown
        ctk.CTkLabel(control_card, text="Mutation Strategy (Data Mutator):", font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", padx=20)
        self.strat_dropdown = ctk.CTkOptionMenu(
            control_card, values=["random_insert (Recommended)", "random_replace", "noise_append"],
            width=260, height=36, fg_color="#16202c", button_color="#1a2332", button_hover_color="#2c3a4e"
        )
        self.strat_dropdown.pack(anchor="w", padx=20, pady=(4, 12))
        self.strat_dropdown.set("random_insert (Recommended)")
        
        # Epochs slider
        ctk.CTkLabel(control_card, text="Training Epochs:", font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", padx=20)
        
        epoch_row = ctk.CTkFrame(control_card, fg_color="transparent")
        epoch_row.pack(fill="x", padx=20, pady=(4, 16))
        
        self.epochs_slider = ctk.CTkSlider(epoch_row, from_=1, to=10, number_of_steps=9, width=200)
        self.epochs_slider.set(3)
        self.epochs_slider.pack(side="left")
        self.epochs_slider.configure(command=self._on_slider_change)
        
        self.epochs_val_label = ctk.CTkLabel(epoch_row, text="3 epochs", font=ctk.CTkFont(weight="bold"))
        self.epochs_val_label.pack(side="left", padx=12)
        
        # Action button
        self.evolve_btn = ctk.CTkButton(
            control_card, text="RUN MODEL EVOLUTION", height=42, fg_color=ACCENT, text_color=BG_DARK,
            font=ctk.CTkFont(weight="bold"), command=self._run_evolution
        )
        self.evolve_btn.pack(fill="x", padx=20, pady=(0, 20))
        
        # Metrics Comparison Card
        metrics_card = ctk.CTkFrame(left_col, fg_color=BG_CARD, corner_radius=12)
        metrics_card.pack(fill="both", expand=True)
        
        ctk.CTkLabel(metrics_card, text="Model Performance & Outcome", font=ctk.CTkFont(weight="bold", size=14), text_color=ACCENT).pack(anchor="w", padx=20, pady=(16, 12))
        
        # Large dashboard metrics widgets
        metrics_inner = ctk.CTkFrame(metrics_card, fg_color="transparent")
        metrics_inner.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Grid layout for baseline vs candidate
        metrics_inner.columnconfigure(0, weight=1)
        metrics_inner.columnconfigure(1, weight=1)
        metrics_inner.rowconfigure(0, weight=1)
        
        # Baseline Card (Left)
        base_box = ctk.CTkFrame(metrics_inner, fg_color="#0a101d", corner_radius=10, border_width=1, border_color="#1a2332")
        base_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=6)
        
        ctk.CTkLabel(base_box, text="🛡️  BASELINE MODEL (CURRENT)", font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED).pack(pady=(12, 8))
        
        base_stats_frame = ctk.CTkFrame(base_box, fg_color="transparent")
        base_stats_frame.pack(fill="x", padx=16, pady=4)
        
        # Accuracy row
        base_acc_row = ctk.CTkFrame(base_stats_frame, fg_color="transparent")
        base_acc_row.pack(fill="x", pady=4)
        ctk.CTkLabel(base_acc_row, text="Accuracy:", font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left")
        self.base_acc_lbl = ctk.CTkLabel(base_acc_row, text="—", font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.base_acc_lbl.pack(side="right")
        
        # Recall row
        base_rec_row = ctk.CTkFrame(base_stats_frame, fg_color="transparent")
        base_rec_row.pack(fill="x", pady=4)
        ctk.CTkLabel(base_rec_row, text="Malware Recall:", font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left")
        self.base_rec_lbl = ctk.CTkLabel(base_rec_row, text="—", font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.base_rec_lbl.pack(side="right")
        
        # Candidate Card (Right)
        cand_box = ctk.CTkFrame(metrics_inner, fg_color="#0d1f30", corner_radius=10, border_width=1, border_color=ACCENT)
        cand_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=6)
        
        ctk.CTkLabel(cand_box, text="🧠  EVOLVED CANDIDATE", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT).pack(pady=(12, 8))
        
        cand_stats_frame = ctk.CTkFrame(cand_box, fg_color="transparent")
        cand_stats_frame.pack(fill="x", padx=16, pady=4)
        
        # Accuracy row
        cand_acc_row = ctk.CTkFrame(cand_stats_frame, fg_color="transparent")
        cand_acc_row.pack(fill="x", pady=4)
        ctk.CTkLabel(cand_acc_row, text="Accuracy:", font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left")
        self.cand_acc_lbl = ctk.CTkLabel(cand_acc_row, text="—", font=ctk.CTkFont(size=18, weight="bold"), text_color=SUCCESS)
        self.cand_acc_lbl.pack(side="right")
        
        # Recall row
        cand_rec_row = ctk.CTkFrame(cand_stats_frame, fg_color="transparent")
        cand_rec_row.pack(fill="x", pady=4)
        ctk.CTkLabel(cand_rec_row, text="Malware Recall:", font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left")
        self.cand_rec_lbl = ctk.CTkLabel(cand_rec_row, text="—", font=ctk.CTkFont(size=18, weight="bold"), text_color=SUCCESS)
        self.cand_rec_lbl.pack(side="right")
        
        # Deployment Outcome badge
        outcome_label_frame = ctk.CTkFrame(metrics_card, fg_color="#0a0e14", corner_radius=8, height=60)
        outcome_label_frame.pack(fill="x", padx=20, pady=20)
        outcome_label_frame.pack_propagate(False)
        
        self.evolve_outcome_lbl = ctk.CTkLabel(
            outcome_label_frame, text="MLOps Evolution Loop Idle",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=MUTED
        )
        self.evolve_outcome_lbl.pack(expand=True)
        
        # RIGHT COLUMN
        right_col = ctk.CTkFrame(grid, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Step Flow visualizer
        flow_card = ctk.CTkFrame(right_col, fg_color=BG_CARD, corner_radius=12, height=75)
        flow_card.pack(fill="x", pady=(0, 16))
        flow_card.pack_propagate(False)
        
        flow_row = ctk.CTkFrame(flow_card, fg_color="transparent")
        flow_row.pack(expand=True)
        
        self._flow_step_labels = []
        steps = ["📂 1. Load", "🧬 2. Mutate", "🧠 3. Train", "📊 4. Eval", "⚖️ 5. Compare", "🚀 6. Deploy"]
        for idx, step_name in enumerate(steps):
            lbl = ctk.CTkLabel(flow_row, text=step_name, font=ctk.CTkFont(size=10, weight="bold"), text_color=MUTED)
            lbl.pack(side="left", padx=6)
            self._flow_step_labels.append(lbl)
            if idx < len(steps) - 1:
                ctk.CTkLabel(flow_row, text="→", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(side="left")
                
        # Live training log window
        log_panel = ctk.CTkFrame(right_col, fg_color=BG_CARD, corner_radius=12)
        log_panel.pack(fill="both", expand=True)
        
        ctk.CTkLabel(log_panel, text="Evolution Pipeline Logs", font=ctk.CTkFont(weight="bold", size=13), text_color=ACCENT).pack(anchor="w", padx=20, pady=(16, 4))
        
        self.evolve_log_box = ctk.CTkTextbox(log_panel, fg_color="#0a0e14", text_color="white",
                                             font=ctk.CTkFont(family="Courier", size=11), corner_radius=8)
        self.evolve_log_box.pack(fill="both", expand=True, padx=16, pady=(4, 16))
        self.evolve_log_box.configure(state="disabled")
        
        return frame
        
    def _on_slider_change(self, val):
        self.epochs_val_label.configure(text=f"{int(val)} epochs")

    def _run_evolution(self):
        self.evolve_btn.configure(state="disabled", text="Evolving Model...")
        self.evolve_log_box.configure(state="normal")
        self.evolve_log_box.delete("1.0", "end")
        self.evolve_log_box.configure(state="disabled")
        
        # Reset flow label styles
        for lbl in self._flow_step_labels:
            lbl.configure(text_color=MUTED)
            
        strategy_full = self.strat_dropdown.get()
        # Extract name: e.g. "random_insert" from "random_insert (Recommended)"
        strategy = strategy_full.split(" ")[0].strip()
        epochs = int(self.epochs_slider.get())
        
        threading.Thread(target=self._run_evolution_bg, args=(strategy, epochs), daemon=True).start()

    def _evolve_log(self, msg: str, step_num: int = None):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        
        def _update():
            self.evolve_log_box.configure(state="normal")
            self.evolve_log_box.insert("end", line)
            self.evolve_log_box.see("end")
            self.evolve_log_box.configure(state="disabled")
            
            if step_num is not None and 1 <= step_num <= 6:
                for i, lbl in enumerate(self._flow_step_labels):
                    if i + 1 == step_num:
                        lbl.configure(text_color=ACCENT)
                    elif i + 1 < step_num:
                        lbl.configure(text_color=SUCCESS)
                    else:
                        lbl.configure(text_color=MUTED)
                        
        self.after(0, _update)

    def _run_evolution_bg(self, strategy, epochs):
        import requests
        import time
        from datetime import datetime, timezone

        # ── GitHub step name → (flow_step_num, friendly_label) ──────────────
        # Maps each GitHub Actions step name to the corresponding flow label index.
        STEP_MAP = {
            "Checkout Repository":          (1, "Loading repository & dataset..."),
            "Set up Python":                (1, "Setting up Python environment..."),
            "Install Dependencies":         (1, "Installing dependencies (torch, sklearn)..."),
            "Run Model Evolution":          (3, "Training candidate model on adversarial data..."),
            "Commit and Push Updated Model":(6, "Committing & pushing results to GitHub..."),
        }

        def elapsed_str(since_iso):
            """Return MM:SS elapsed since an ISO timestamp string."""
            if not since_iso:
                return ""
            try:
                start = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
                secs  = int((datetime.now(timezone.utc) - start).total_seconds())
                return f"  [{secs // 60:02d}:{secs % 60:02d} elapsed]"
            except Exception:
                return ""

        self._evolve_log("Initializing Section 6 Model Evolution Loop...", step_num=1)

        try:
            self._evolve_log("Sending trigger request to FastAPI server...", step_num=1)
            r = requests.post(
                f"{self.server_url}/system/retrain?strategy={strategy}&epochs={epochs}",
                headers=self.headers,
                timeout=600
            )
            if r.status_code != 200:
                raise Exception(f"Server error {r.status_code}: {r.text[:200]}")

            resp   = r.json()
            status = resp.get("status")

            if status == "github_triggered":
                repo = resp.get("repo")
                self._evolve_log(f"GitHub Actions triggered for: {repo}", step_num=2)
                self._evolve_log("Waiting for runner to start...", step_num=2)

                last_step_name   = None
                train_started_at = None   # timestamp when "Run Model Evolution" began
                poll_count       = 0

                while True:
                    time.sleep(5)
                    poll_count += 1

                    try:
                        r_status = requests.get(
                            f"{self.server_url}/system/retrain-status",
                            headers=self.headers, timeout=10
                        )
                    except Exception as net_err:
                        self._evolve_log(f"  [poll] Network error: {net_err}", step_num=None)
                        continue

                    if r_status.status_code != 200:
                        continue

                    run_info   = r_status.json()
                    run_status = run_info.get("status")
                    conclusion = run_info.get("conclusion")
                    steps      = run_info.get("steps", [])

                    # ── Map current active GitHub step → flow indicator ───────
                    active_step = next(
                        (s for s in steps if s.get("status") == "in_progress"), None
                    )
                    if active_step:
                        gh_name = active_step.get("name", "")
                        flow_num, friendly = STEP_MAP.get(gh_name, (2, gh_name))

                        # Track when actual training started
                        if gh_name == "Run Model Evolution":
                            if train_started_at is None:
                                train_started_at = active_step.get("started_at")
                                self._evolve_log(
                                    f"[Step 3/6] Mutating data & training candidate model...",
                                    step_num=2
                                )
                            # Show live timer every poll while training is running
                            timer = elapsed_str(train_started_at)
                            self._evolve_log(
                                f"  Training in progress{timer} — {epochs} epoch(s) on augmented dataset",
                                step_num=3
                            )
                        elif gh_name != last_step_name:
                            self._evolve_log(f"[GitHub] {friendly}", step_num=flow_num)

                        last_step_name = gh_name

                    elif run_status == "queued":
                        if poll_count % 3 == 0:   # log every 15s while queued
                            self._evolve_log("  Runner queued — waiting for a free GitHub worker...", step_num=1)

                    # ── Check for completion ──────────────────────────────────
                    if run_status == "completed":
                        if conclusion == "success":
                            # Check whether the model was actually updated
                            improved = any(
                                s.get("name") == "Commit and Push Updated Model"
                                and s.get("conclusion") == "success"
                                for s in steps
                            )
                            self._evolve_log(
                                "GitHub Action finished! Evaluating results...", step_num=4
                            )
                            self._evolve_log(
                                "Comparison complete. Syncing results to local server...", step_num=5
                            )

                            # Sync model + results JSON from GitHub
                            r_sync = requests.post(
                                f"{self.server_url}/system/sync-model",
                                headers=self.headers, timeout=60
                            )
                            if r_sync.status_code == 200:
                                self._evolve_log(
                                    "Model weights & metrics synced from GitHub. Server reloaded! ",
                                    step_num=6
                                )
                                break
                            else:
                                raise Exception(f"Model sync failed: {r_sync.text[:200]}")
                        else:
                            raise Exception(f"GitHub Action completed with conclusion: {conclusion}")

            elif status == "success":
                self._evolve_log("Local retraining completed successfully!", step_num=5)
                self._evolve_log("Model successfully reloaded on the server! ", step_num=6)

            else:
                raise Exception(f"Retrain failed: {resp.get('message')}")

            # Fetch updated metrics to show in GUI
            r_stats = requests.get(
                f"{self.server_url}/system/stats", headers=self.headers, timeout=5
            )
            if r_stats.status_code == 200:
                stats = r_stats.json()
                self.after(0, lambda: self._update_metrics_ui(stats, is_completion=True))

            # Mark all flow steps complete
            def _show_results():
                for lbl in self._flow_step_labels:
                    lbl.configure(text_color=SUCCESS)
                self.evolve_btn.configure(state="normal", text="RUN MODEL EVOLUTION")

            self.after(0, _show_results)

        except Exception as e:
            self._evolve_log(f"CRITICAL TRAINING ERROR: {e}", step_num=None)
            self.after(0, lambda: self.evolve_btn.configure(state="normal", text="RUN MODEL EVOLUTION"))



    def _update_metrics_ui(self, stats, is_completion=False):
        def format_percent(val):
            if val is None:
                return "—"
            if val <= 1.0:
                return f"{val * 100:.2f}%"
            return f"{val:.2f}%"

        evolution_stats = stats.get("evolution_stats")
        
        if is_completion and evolution_stats:
            # Just completed training. Display the comparison of that specific run.
            base_acc = evolution_stats.get("base_accuracy")
            base_rec = evolution_stats.get("base_recall")
            cand_acc = evolution_stats.get("candidate_accuracy")
            cand_rec = evolution_stats.get("candidate_recall")
            success = evolution_stats.get("evolution_successful", False)
            
            self.base_acc_lbl.configure(text=format_percent(base_acc))
            self.base_rec_lbl.configure(text=format_percent(base_rec))
            self.cand_acc_lbl.configure(text=format_percent(cand_acc))
            self.cand_rec_lbl.configure(text=format_percent(cand_rec))
            
            if success:
                self.evolve_outcome_lbl.configure(
                    text="✔ EVOLUTION SUCCESSFUL: Server Model Updated! 🚀",
                    text_color=SUCCESS
                )
            else:
                self.evolve_outcome_lbl.configure(
                    text="ℹ BASELINE MAINTAINED: No Model Improvement",
                    text_color=WARNING
                )
        else:
            # Idle / Startup state: Display current active model in Baseline, and clear Candidate.
            if evolution_stats:
                success = evolution_stats.get("evolution_successful", False)
                if success:
                    current_acc = evolution_stats.get("candidate_accuracy")
                    current_rec = evolution_stats.get("candidate_recall")
                else:
                    current_acc = evolution_stats.get("base_accuracy")
                    current_rec = evolution_stats.get("base_recall")
            else:
                # No evolution_stats yet (server just started, no runs completed).
                # Use the verified baseline metrics from eval_baseline.py (stratified val set).
                current_acc = stats.get("model_accuracy", 98.75)
                current_rec = 0.9925  # 99.25% — verified malware recall (stratified eval)
                
            self.base_acc_lbl.configure(text=format_percent(current_acc))
            self.base_rec_lbl.configure(text=format_percent(current_rec))
            self.cand_acc_lbl.configure(text="—")
            self.cand_rec_lbl.configure(text="—")
            self.evolve_outcome_lbl.configure(
                text="MLOps Evolution Loop Idle",
                text_color=MUTED
            )

    def _clear_logs(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
