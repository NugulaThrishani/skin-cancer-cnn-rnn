"""Simple test script that POSTs an image file to the running Flask server's /predict endpoint.

Usage:
  python test_predict.py path/to/image.jpg
"""
import sys
import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_predict.py path/to/image.jpg")
        return
    img_path = sys.argv[1]
    url = "http://127.0.0.1:5000/predict"
    with open(img_path, "rb") as f:
        files = {"image": f}
        r = requests.post(url, files=files)
        try:
            print(r.status_code)
            print(r.json())
        except Exception:
            print(r.text)

if __name__ == "__main__":
    main()
