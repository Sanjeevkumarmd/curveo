"""
Curveo agent — generate images & videos from prompts.

Primary provider: DeepInfra (images + video)
Optional: Google Gemini for scene planning (if GEMINI_API_KEY is set)

API keys live only in local .env (never commit them).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DEEPINFRA_BASE = "https://api.deepinfra.com"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def deepinfra_key() -> str:
    load_env()
    key = (os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPINFRA_TOKEN") or "").strip()
    if not key or key.startswith("YOUR_"):
        print(
            "Missing DeepInfra key.\n"
            "Set DEEPINFRA_API_KEY=... in .env\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def with_audio_direction(prompt: str) -> str:
    lower = prompt.lower()
    if any(w in lower for w in ("sound", "audio", "music", "ambient", "sfx", "dialogue")):
        return prompt
    return (
        f"{prompt.strip()} "
        "Include rich soundtrack cues: realistic ambient sound, "
        "synchronized foley, soft cinematic music. "
        "Keep dialogue only if it fits naturally."
    )


def plan_scenes(prompt: str, target_sec: int, clip_sec: int) -> list[str]:
    """Split a long brief into scene prompts. Uses Gemini if available, else simple split."""
    n = max(1, (target_sec + clip_sec - 1) // clip_sec)
    load_env()
    gkey = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()

    if gkey and not gkey.startswith("YOUR_"):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gkey)
            planner_prompt = f"""You are a film director creating a continuous video WITH SOUND.
Split this brief into exactly {n} consecutive scene prompts.
Each scene is ~{clip_sec} seconds and MUST describe visuals and audio.
Return ONLY a JSON array of {n} strings. No markdown.

Brief:
{prompt}
"""
            response = client.models.generate_content(
                model=os.getenv("PLANNER_MODEL", "gemini-3.6-flash"),
                contents=planner_prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            scenes = json.loads(text)
            if isinstance(scenes, list) and scenes:
                return [with_audio_direction(str(s).strip()) for s in scenes][:n]
        except Exception as exc:
            print(f"Planner fallback (Gemini unavailable: {exc})")

    # Simple fallback: repeat enriched prompt
    return [with_audio_direction(prompt)] * n


def generate_image(prompt: str, out_path: Path) -> Path:
    key = deepinfra_key()
    model = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX-1-schnell")
    print(f"Generating image with DeepInfra / {model}...")
    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            f"{DEEPINFRA_BASE}/v1/openai/images/generations",
            headers=headers(key),
            json={
                "model": model,
                "prompt": prompt,
                "size": os.getenv("IMAGE_SIZE", "1024x1024"),
                "n": 1,
                "response_format": "b64_json",
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"Image failed ({r.status_code}): {r.text[:500]}")
        b64 = r.json()["data"][0]["b64_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64))
    print(f"Saved image: {out_path}")
    return out_path


def _download(url: str, out_path: Path) -> Path:
    if url.startswith("/"):
        url = DEEPINFRA_BASE.rstrip("/") + url
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
    return out_path


def generate_one_clip(
    prompt: str,
    out_path: Path,
    *,
    image_path: Path | None = None,
    duration_sec: int = 5,
) -> Path:
    key = deepinfra_key()
    prompt = with_audio_direction(prompt)

    if image_path and image_path.exists():
        model = os.getenv("VIDEO_I2V_MODEL", "Pixverse/Pixverse-6-I2V")
    else:
        model = os.getenv("VIDEO_MODEL", "Pixverse/Pixverse-T2V")

    # Pixverse-T2V allows 5 or 8
    duration_sec = 8 if duration_sec >= 8 else 5
    print(f"Generating clip ({duration_sec}s) with DeepInfra / {model}...")
    print(f"  Prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}")

    payload: dict = {
        "prompt": prompt,
        "duration": duration_sec,
        "aspect_ratio": os.getenv("VIDEO_ASPECT", "16:9"),
    }

    # Some I2V models want an image URL; upload not always available —
    # for local images, use T2V with strong prompt unless HTTP URL given.
    if image_path and str(image_path).startswith("http"):
        payload["image"] = str(image_path)
        payload["image_url"] = str(image_path)

    with httpx.Client(timeout=None) as client:
        r = client.post(
            f"{DEEPINFRA_BASE}/v1/inference/{model}",
            headers=headers(key),
            json=payload,
            timeout=600.0,
        )
        if r.status_code != 200:
            # Retry without duration if model rejects it
            if "duration" in r.text.lower() or r.status_code == 422:
                payload.pop("duration", None)
                r = client.post(
                    f"{DEEPINFRA_BASE}/v1/inference/{model}",
                    headers=headers(key),
                    json={"prompt": prompt, "aspect_ratio": payload.get("aspect_ratio", "16:9")},
                    timeout=600.0,
                )
        if r.status_code != 200:
            raise RuntimeError(f"Video failed ({r.status_code}): {r.text[:800]}")

        data = r.json()
        video_url = data.get("video_url") or data.get("output") or data.get("video")
        if isinstance(video_url, list):
            video_url = video_url[0] if video_url else None
        if not video_url:
            raise RuntimeError(f"No video_url in response: {json.dumps(data)[:500]}")

        _download(video_url, out_path)

    print(f"  Saved clip: {out_path}")
    return out_path


def stitch_clips(clips: list[Path], out_path: Path) -> Path:
    if len(clips) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(clips[0], out_path)
        return out_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "ffmpeg not found — left individual clips in output/. "
            "Install ffmpeg to merge into one file.",
            file=sys.stderr,
        )
        return clips[0]

    list_file = out_path.parent / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out_path),
    ]
    print("Stitching clips with ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"Final video: {out_path}")
    return out_path


def cmd_image(args: argparse.Namespace) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else OUTPUT / f"image-{stamp}.png"
    generate_image(args.prompt, out)


def cmd_video(args: argparse.Namespace) -> None:
    target = int(args.duration)
    clip_sec = 8 if int(args.clip_duration) >= 8 else 5

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = OUTPUT / f"run-{stamp}"
    clips_dir = work / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image) if args.image else None
    if args.make_image and not image_path:
        image_path = work / "reference.png"
        generate_image(args.prompt, image_path)

    print(f"Planning ~{target}s storyboard (clips of {clip_sec}s)...")
    scenes = plan_scenes(args.prompt, target, clip_sec)
    (work / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print(f"Planned {len(scenes)} scenes.")

    clips: list[Path] = []
    for i, scene in enumerate(scenes, start=1):
        clip_path = clips_dir / f"scene-{i:02d}.mp4"
        try:
            generate_one_clip(
                scene,
                clip_path,
                image_path=image_path if i == 1 else None,
                duration_sec=clip_sec,
            )
            clips.append(clip_path)
        except Exception as exc:
            print(f"Scene {i} failed: {exc}", file=sys.stderr)
            if not clips:
                raise
            print("Continuing with clips generated so far...")
            break

    if not clips:
        raise RuntimeError("No clips were generated.")

    final = Path(args.out) if args.out else work / f"curveo-{target}s.mp4"
    stitch_clips(clips, final)
    print("\nDone.")
    print(f"Output folder: {work}")
    print(f"Final file:    {final}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curveo — DeepInfra image & video agent")
    sub = p.add_subparsers(dest="command", required=True)

    img = sub.add_parser("image", help="Generate an image from a prompt")
    img.add_argument("prompt", help="Image prompt")
    img.add_argument("--out", help="Output PNG path")
    img.set_defaults(func=cmd_image)

    vid = sub.add_parser("video", help="Generate video (plans + clips + stitch)")
    vid.add_argument("prompt", help="Video prompt / creative brief")
    vid.add_argument("--image", help="Optional reference image path or URL")
    vid.add_argument("--make-image", action="store_true", help="Generate reference image first")
    vid.add_argument("--duration", type=int, default=120, help="Target length seconds (default 120)")
    vid.add_argument("--clip-duration", type=int, default=5, help="Seconds per clip: 5 or 8")
    vid.add_argument("--out", help="Final MP4 path")
    vid.set_defaults(func=cmd_video)

    return p


def main() -> None:
    load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
