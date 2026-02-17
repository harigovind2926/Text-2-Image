🚀 Real-Time Text-to-Image AI

A real-time Text-to-Image generation system built using Stable Diffusion v1.5 with LCM LoRA acceleration, designed for fast, safe, and realistic image generation.

📌 Project Overview

This project converts natural language text prompts into realistic images using diffusion-based generative AI models. The system is optimized for near real-time performance (≈1–2 seconds on GPU) and includes strict content control to restrict human-related and sensitive outputs.

✨ Features

⚡ Real-time image generation (LCM optimized)

🧠 Prompt analysis and enhancement

🎨 Automatic realism injection

🚫 Human content restriction

🔒 Sensitive content filtering

🖥️ Clean frontend using Gradio

🔌 Backend API built with FastAPI

📂 Batch testing with prompt runner

💾 Organized output folder saving

🏗️ System Architecture
Frontend (Gradio UI)
        ↓
FastAPI Backend (/generate)
        ↓
Prompt Validation & Enhancement
        ↓
Stable Diffusion + LCM LoRA
        ↓
Image → Base64 → Frontend

🛠️ Technologies Used

Python 3.10+

FastAPI

Gradio

PyTorch

Hugging Face Diffusers

Stable Diffusion v1.5

LCM LoRA

CUDA (for GPU acceleration)

💻 System Requirements
Minimum

Windows 10/11 (64-bit)

16 GB RAM

Python 3.10+

CPU (for testing only)

Recommended

32 GB RAM

NVIDIA RTX GPU (8GB+ VRAM)

CUDA 11.8

SSD storage

📦 Installation Guide
1️⃣ Clone or Download the Project

Navigate to the project directory.

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install CUDA PyTorch (For GPU Systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

5️⃣ Run Backend Server
cd Backend
uvicorn main:app --host 127.0.0.1 --port 8000

6️⃣ Run Frontend UI

Open a new terminal:

cd Frontend
python ui.py


The browser will open automatically.

🧪 Batch Testing (Optional)

To test multiple prompts:

Add prompts (one per line) in:

prompts.txt


Run:

python prompt_runner.py


Images will be saved inside:

outputs/run_timestamp/

🔐 Safety Controls

The system blocks:

Human-related prompts (man, woman, people, portrait, etc.)

Sensitive content (blood, violence, nude, nsfw, etc.)

All prompts are validated before image generation.

⚡ Performance Optimization

Acceleration is achieved using:

Stable Diffusion v1.5

LCM LoRA adapter

LCMScheduler

Reduced inference steps (≈5)

Low guidance scale

Result:

~1–2 seconds per image on RTX GPU

📂 Project Structure
AI/
│
├── Backend/
│   └── main.py
│
├── Frontend/
│   └── ui.py
│
├── prompt_runner.py
├── prompts.txt
├── requirements.txt
└── README.md

📊 Testing & Validation

200+ prompt dataset

Sequential execution testing

Stability validation

Safety verification

🚀 Future Enhancements

High-resolution generation

Style selector (realistic / artistic)

Multi-image output

Image editing features

Cloud deployment

📄 License

This project is developed for educational and research purposes.

👨‍💻 Author

Developed as part of a Generative AI project focusing on real-time image synthesis and system optimization.
