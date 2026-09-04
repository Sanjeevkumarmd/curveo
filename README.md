# Curveo

### Turn a text prompt into images & videos **with sound** — powered by Google Veo + Imagen

[![GitHub](https://img.shields.io/badge/repo-Sanjeevkumarmd%2Fcurveo-0b1c22?style=flat-square)](https://github.com/Sanjeevkumarmd/curveo)
[![Python](https://img.shields.io/badge/python-3.10%2B-3ecf9a?style=flat-square)](#setup)
[![API](https://img.shields.io/badge/Google-Gemini%20%7C%20Veo%20%7C%20Imagen-4db8e8?style=flat-square)](https://aistudio.google.com/apikey)

Curveo is a small local agent: you describe a scene, it plans a storyboard, generates **short Veo clips with native audio** (ambience, foley, music, optional dialogue), then stitches them toward a **~2 minute** film.

> **Security:** never commit your API key. Keep it only in a local `.env` file.

---

## What you can make

| Command | Result |
|--------|--------|
| `python agent.py image "..."` | Still image (Imagen) |
| `python agent.py video "..."` | Video **with sound** (~2 min by default) |
| `python agent.py video "..." --make-image` | Image first, then video guided by that look |
| `python agent.py video "..." --image photo.png` | Animate / continue from your own image |

**Sound tip:** describe audio in the prompt, e.g.  
`"workshop ambience, soft synth music, metal tools clinking, no dialogue"`.

Veo 3.1 generates **picture + sound together**. Curveo also injects audio direction into every scene so clips aren’t silent by accident.

---

## Quick start (anyone can follow)

### 1. Clone

```bash
git clone https://github.com/Sanjeevkumarmd/curveo.git
cd curveo
```

### 2. Add your Google API key

1. Create a key: [Google AI Studio → API keys](https://aistudio.google.com/apikey)
2. Enable **billing** (Veo / Imagen usually need a paid project)
3. Copy the example env file:

**Windows (PowerShell)**
```powershell
copy .env.example .env
notepad .env
```

**macOS / Linux**
```bash
cp .env.example .env
nano .env
```

4. Paste your key:

```env
GEMINI_API_KEY=your_key_here
```

That is the **only** place to add or change the key.

### 3. Install

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Install [ffmpeg](https://ffmpeg.org/download.html) and add it to your PATH so scenes merge into one MP4 with continuous audio.

### 4. Create

**Image**
```bash
python agent.py image "Engineering student at a glowing CAD workstation, cinematic light"
```

**~2-minute video with sound**
```bash
python agent.py video "A cinematic 2-minute tour of a university makerspace: robots, 3D printers, soft circuits. Soft electronic music, workshop ambience, tool sounds, short friendly VO optional." --duration 120
```

**Image + video**
```bash
python agent.py video "Same makerspace tour, warm daylight, smooth camera, rich ambient audio" --make-image --duration 120
```

Outputs appear under `output/run-TIMESTAMP/`.

---

## How a 2-minute video works

Google Veo renders **short clips** (about 4–8 seconds), not one long 120s shot.

Curveo:

1. **Plans** your brief into timed scene prompts (visual + audio)
2. **Generates** each scene with Veo (native soundtrack)
3. **Stitches** clips with ffmpeg toward `--duration` (default `120`)

A full 2-minute run can take **many minutes** and use **many API calls** — start with `--duration 8` to test.

---

## Change the API key later

1. Open `.env`
2. Replace `GEMINI_API_KEY=...`
3. Save — no code edits needed

Optional settings in `.env`:

```env
VIDEO_MODEL=veo-3.1-generate-preview
IMAGE_MODEL=gemini-3.1-flash-image
PLANNER_MODEL=gemini-3.6-flash
TARGET_DURATION_SEC=120
CLIP_DURATION_SEC=8
```

---

## Project layout

```
curveo/
├── agent.py          # CLI agent (image + video + sound)
├── .env.example      # Template for your key (safe to commit)
├── .env              # Your real key (gitignored — never push)
├── requirements.txt
├── README.md
└── output/           # Generated media (gitignored)
```

---

## Troubleshooting

| Issue | Fix |
|------|-----|
| `Missing API key` | Create `.env` from `.env.example` |
| 403 / permission errors | Enable Veo/Imagen + billing on the Google project |
| Silent video | Mention sounds/music/ambience in the prompt |
| Many separate clips, no final MP4 | Install ffmpeg |
| Slow / expensive | Test with `--duration 8` first |

---

## License

Use freely for learning and demos. You are responsible for Google API usage, billing, and content policy compliance.

Built for creators who want **prompt → video with sound** without a heavy UI.
