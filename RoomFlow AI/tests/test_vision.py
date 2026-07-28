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

url = get_setting("llm_url", MODELS["llm"]["base_url"])
model = MODELS["llm"]["model"]
print(f"URL:   {url}")
print(f"Model: {model}")
print(f"Image: {img_path}")

resp = requests.post(
    f"{url}/chat/completions",
    json={
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this person briefly in 1 sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 100,
    },
    timeout=60,
)
print(f"\nStatus: {resp.status_code}")
print(f"Response:\n{json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
