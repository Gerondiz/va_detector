import requests, base64, sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.database import get_setting
from backend.config import MODELS

img_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not img_path or not os.path.exists(img_path):
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    jpgs = sorted([f for f in os.listdir(logs_dir) if f.endswith(".jpg")])
    if not jpgs:
        print("no images found — pass path as argument")
        sys.exit(1)
    img_path = os.path.join(logs_dir, jpgs[0])

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_url = f"data:image/jpeg;base64,{b64}"

url = get_setting("llm_url", MODELS["llm"]["base_url"])
print(f"URL:  {url}")
print(f"Img:  {img_path}")

# Test 1 — moondream (vision)
vm = get_setting("vision_model", "moondream:latest")
print(f"\n--- moondream ({vm}) ---")
r1 = requests.post(f"{url}/chat/completions", json={
    "model": vm,
    "messages": [{"role":"user","content":[
        {"type":"text","text":"Describe this person briefly in 1 sentence."},
        {"type":"image_url","image_url":{"url":data_url}},
    ]}],
    "max_tokens": 200,
}, timeout=60)
t1 = r1.json()["choices"][0]["message"]["content"].strip()
print(f"Status: {r1.status_code}, response: {t1[:200]}")

# Test 2 — gemma4 (text only, no image)
lm = MODELS["llm"]["model"]
print(f"\n--- gemma4 ({lm}) ---")
r2 = requests.post(f"{url}/chat/completions", json={
    "model": lm,
    "messages": [{"role":"user","content":f'A person was described as: "{t1}". Does this match ID 1 described as "tall man in blue jacket"? Return JSON {{"person_id": <int>}}'}],
    "max_tokens": 100,
}, timeout=30)
print(f"Status: {r2.status_code}")
print(f"Response:\n{json.dumps(r2.json(), indent=2, ensure_ascii=False)[:500]}")
