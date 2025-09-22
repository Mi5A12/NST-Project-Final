#!/usr/bin/env python3
"""
Simple script to run the Streamlit Neural Style Transfer application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app with proper configuration"""
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if required files exist
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found")
        sys.exit(1)
    
    if not os.path.exists("neural_style_transfer.py"):
        print("❌ neural_style_transfer.py not found")
        sys.exit(1)
    
    print("🚀 Starting Neural Style Transfer Application...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
