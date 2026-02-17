import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, LCMScheduler
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# -----------------------------
# MODEL + LCM LoRA
# -----------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
).to(device)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)

# -----------------------------
# CONTROL LISTS
# -----------------------------
ANIMALS = [
    "tiger", "lion", "leopard", "cheetah", "wolf",
    "dog", "cat", "elephant", "horse", "bear", "deer"
]

MONUMENTS = [
    "taj mahal", "eiffel tower", "statue of liberty",
    "colosseum", "great wall", "machu picchu", "petra"
]

WEATHER_MAP = {
    "snow": "snowfall, winter atmosphere, snow-covered ground",
    "rain": "rainy weather, wet surfaces, overcast sky",
    "fog": "foggy atmosphere, low visibility",
    "night": "night time lighting, artificial lights",
    "sunset": "sunset lighting, warm tones",
    "storm": "stormy sky"
}

# 🚫 HARD HUMAN BLOCK
HUMAN_KEYWORDS = [
    "human", "person", "people", "man", "woman", "boy", "girl",
    "child", "children", "face", "faces", "body", "bodies",
    "portrait", "selfie", "crowd", "group", "silhouette",
    "actor", "actress", "model", "soldier", "worker"
]

# 🚫 SENSITIVE CONTENT BLOCK
BANNED_WORDS = [
    "blood", "gore", "dead", "kill", "violence",
    "nude", "naked", "sex", "explicit", "nsfw"
]

# -----------------------------
# PROMPT BASE (ALWAYS APPLIED)
# -----------------------------
REALISM_BASE = (
    "ultra realistic photo, DSLR photography, natural lighting, "
    "sharp focus, high detail, accurate proportions"
)

SCENE_BIAS = "scene without people, empty environment"

NEGATIVE_PROMPT = (
    "people, person, humans, human, man, woman, boy, girl, child, children, "
    "face, faces, head, body, bodies, silhouette, crowd, group, portrait, selfie, "
    "skin, hands, fingers, arms, legs, feet, "
    "blood, gore, violence, dead, kill, injury, "
    "nude, naked, sexual, explicit, nsfw, "
    "cartoon, anime, illustration, cgi, 3d render, "
    "distorted, deformed, extra limbs, blurry"
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Prompt(BaseModel):
    text: str

# -----------------------------
# GENERATE ENDPOINT (FINAL)
# -----------------------------
@app.post("/generate")
def generate_image(data: Prompt):
    try:
        user_prompt = data.text.lower().strip()

        # 🚫 BLOCK HUMAN PROMPTS
        for word in HUMAN_KEYWORDS:
            if word in user_prompt:
                return {"error": "Human-related prompts are not allowed"}

        # 🚫 BLOCK SENSITIVE PROMPTS
        for bad in BANNED_WORDS:
            if bad in user_prompt:
                return {"error": "Unsafe prompt blocked"}

        enhancements = []

        # 🌦️ Weather enhancement
        for key, value in WEATHER_MAP.items():
            if key in user_prompt:
                enhancements.append(value)
                break

        # 🐾 Animal realism
        for animal in ANIMALS:
            if animal in user_prompt:
                enhancements.append(
                    "realistic wildlife photography, natural habitat"
                )
                break

        # 🏛️ Monument realism
        for monument in MONUMENTS:
            if monument in user_prompt:
                enhancements.append(
                    "ground level view, front view, realistic architecture photography"
                )
                break

        # 🔗 FINAL PROMPT (USER PRESERVED)
        final_prompt = (
            f"{user_prompt}, {REALISM_BASE}, {SCENE_BIAS}"
            + (", " + ", ".join(enhancements) if enhancements else "")
        )

        with torch.inference_mode(), torch.autocast("cuda"):
            image = pipe(
                final_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=5,   # ⚡ LCM FAST
                guidance_scale=1.0,
                height=512,
                width=512
            ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        return {
            "image": base64.b64encode(buffer.getvalue()).decode()
        }

    except Exception as e:
        print("Backend error:", e)
        return {"error": "generation_failed"}
