import os
from pathlib import Path

os.environ["GEMINI_API_KEY"] = (Path(__file__).resolve().parents[1] / "key.env").read_text(encoding="utf-8").strip()
from google import genai
from google.genai import types

client = genai.Client()
for m in [
    "gemini-3-pro-preview",
    "gemini-3-pro",
    "gemini-3-flash-preview",
    "gemini-3-flash",
    "gemini-3-flash-lite",
    "gemini-3-pro-experimental",
    "gemini-3.0-pro",
    "gemini-3.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]:
    try:
        r = client.models.generate_content(
            model=m,
            contents="reply OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        out = (r.text or "").strip()
        print(f"OK    {m}: {out!r}")
    except Exception as e:
        msg = str(e)[:160].replace("\n", " ")
        print(f"FAIL  {m}: {msg}")
