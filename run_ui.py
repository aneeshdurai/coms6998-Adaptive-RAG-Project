#!/usr/bin/env python3
"""
Simple script to run the Streamlit UI for Adaptive RAG
Automatically uses venv Python if available
"""
import subprocess
import sys
from pathlib import Path

def find_venv_python():
    """Find Python executable in venv, or use current Python"""
    venv_path = Path(__file__).parent / "venv"
    if venv_path.exists():
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        if python_exe.exists():
            return str(python_exe)
    return sys.executable

def main():
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    # Use venv Python if available
    python_exe = find_venv_python()
    
    print("Starting Adaptive RAG UI...")
    print(f"Using Python: {python_exe}")
    print("The UI will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            python_exe, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down UI server...")
    except Exception as e:
        print(f"Error running UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

