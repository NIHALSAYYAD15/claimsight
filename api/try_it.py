import sys
import requests

path = sys.argv[1] if len(sys.argv) > 1 else "data/claims_v2/CLM2-001.txt"
text = open(path, encoding="utf-8").read()
r = requests.post("http://127.0.0.1:8000/claims/analyze", json={"text": text})
print(r.status_code)
print(r.text)