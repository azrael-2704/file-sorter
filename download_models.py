"""Download required models for offline use.

Run this script once after installing dependencies to download models:
python download_models.py
"""

from local_ai import download_models

if __name__ == "__main__":
    print("Downloading required models for offline use...")
    download_models()
    print("Done! The system is ready for offline use.")