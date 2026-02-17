import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO
import time

BACKEND_URL = "http://127.0.0.1:8000/generate"

# -----------------------------
# MANUAL DEBOUNCE
# -----------------------------
last_call_time = 0
DEBOUNCE_SECONDS = 0.8

def generate_from_api(prompt):
    global last_call_time

    if not prompt or not prompt.strip():
        return None

    now = time.time()
    if now - last_call_time < DEBOUNCE_SECONDS:
        return gr.update()

    last_call_time = now

    try:
        response = requests.post(
            BACKEND_URL,
            json={"text": prompt},
            timeout=120
        )

        if response.status_code != 200:
            return None

        data = response.json()

        # 🚫 Blocked or failed generation
        if not data or "image" not in data:
            return None

        img_bytes = base64.b64decode(data["image"])
        return Image.open(BytesIO(img_bytes))

    except Exception as e:
        print("Frontend error:", e)
        return None

# -----------------------------
# UI LAYOUT
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        ## ⚡ Real-Time Text-to-Image AI  
        Type a short prompt → stop typing → image generates automatically  
        """
    )

    prompt = gr.Textbox(
        label="Prompt",
        placeholder="tiger | lion | mountain | Taj Mahal",
        lines=2,
        max_lines=3
    )

    output = gr.Image(
        label="Generated Image",
        height=480
    )

    prompt.change(
        fn=generate_from_api,
        inputs=prompt,
        outputs=output
    )

# -----------------------------
# LAUNCH
# -----------------------------
demo.launch(
    inbrowser=True,
    show_api=False,
    max_threads=1
)
