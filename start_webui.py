#!/usr/bin/env python3
"""
T-BLIP2 Web UI One-Click Launcher
Automatically starts the backend server and opens the browser
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """Check if necessary dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'pillow', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pillow':
                # Special handling for pillow (PIL)
                import PIL
                print(f"✅ {package} (PIL {PIL.__version__}) is available")
            else:
                __import__(package)
                print(f"✅ {package} is available")
        except ImportError as e:
            missing_packages.append(package)
            print(f"❌ {package} import failed: {e}")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please run the following command to install:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages are installed")
    return True

def start_backend_server():
    """Start the backend server"""
    backend_dir = Path(__file__).parent / "backend"
    
    if not backend_dir.exists():
        print(f"❌ Backend directory does not exist: {backend_dir}")
        return False
    
    # Switch to backend directory
    os.chdir(backend_dir)
    
    # Start the server
    print("🚀 Starting T-BLIP2 backend server...")
    try:
        # Use uvicorn to start the server
        print("🔧 Starting uvicorn server...")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Check if server process is still running
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            print(f"❌ Server process terminated unexpectedly")
            print(f"Exit code: {process.returncode}")
            if stdout:
                print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
            return None
        
        # Check if server started successfully
        try:
            import requests
            print("🔍 Checking server status...")
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend server started successfully!")
                return process
            else:
                print("❌ Backend server failed to start")
                return None
        except ImportError:
            # If requests library is not available, assume success
            print("✅ Backend server is starting... (requests not available)")
            return process
        except Exception as e:
            print(f"⚠️ Server status check failed: {e}")
            print("✅ Assuming server is starting...")
            return process
            
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return None

def open_browser():
    """Open the browser"""
    time.sleep(3)  # Wait for server to fully start
    print("🌐 Opening browser...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print("Please manually visit: http://localhost:8000")

def main():
    """Main function"""
    print("=" * 50)
    print("🔍 T-BLIP2 Web UI Launcher")
    print("=" * 50)
    
    # Show environment info
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🔍 Python executable: {sys.executable}")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Start backend server
    server_process = start_backend_server()
    if not server_process:
        print("❌ Unable to start backend server")
        return
    
    # Only open browser if server is actually running
    if server_process:
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    else:
        print("❌ Cannot open browser - server failed to start")
        return
    
    print("\n" + "=" * 50)
    print("🎉 T-BLIP2 Web UI is now running!")
    print("📱 Access URL: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Keep server running
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
