"""
Curveo agent — generate images and videos WITH SOUND from prompts
using Google Gemini / Veo / Imagen.

API key lives only in local .env (never commit it).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_AI_API_KEY")
        or ""
    ).strip()
    if not key or key.startswith("YOUR_"):
        print(
            "Missing API key.\n"
            "1) Copy .env.example → .env\n"
            "2) Set GEMINI_API_KEY=...\n"
            "3) Run again\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def get_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def with_audio_direction(prompt: str) -> str:
    """Ensure every scene prompt asks for synced native audio."""
    lower = prompt.lower()
    if any(w in lower for w in ("sound", "audio", "music", "ambient", "sfx", "dialogue")):
        return prompt
    return (
        f"{prompt.strip()} "
        "Include rich native soundtrack: realistic ambient sound, "
        "synchronized foley for on-screen actions, and soft cinematic music. "
        "Keep dialogue only if it fits the scene naturally."
    )


def plan_scenes(client, prompt: str, target_sec: int, clip_sec: int) -> list[str]:
    from google.genai import types

    n = max(1, (target_sec + clip_sec - 1) // clip_sec)
    planner_prompt = f"""You are a film director creating a continuous video WITH SOUND.
Split this brief into exactly {n} consecutive scene prompts for Google Veo.
Each scene is ~{clip_sec} seconds and MUST describe:
- visuals (camera, subject, lighting, motion)
- audio (ambient, foley, music, optional short dialogue)
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
    try:
        scenes = json.loads(text)
        if isinstance(scenes, list) and scenes:
            return [with_audio_direction(str(s).strip()) for s in scenes][:n]
    except json.JSONDecodeError:
        pass
    return [with_audio_direction(prompt)] * n


def generate_image(client, prompt: str, out_path: Path) -> Path:
    from google.genai import types

    model = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")
    print(f"Generating image with {model}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer Gemini native image models (available on many API keys)
    try:
        result = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        for part in result.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                out_path.write_bytes(inline.data)
                print(f"Saved image: {out_path}")
                return out_path
        raise RuntimeError("Model returned no image bytes.")
    except Exception as primary_err:
        # Fallback to classic Imagen if configured / available
        try:
            result = client.models.generate_images(
                model=os.getenv("IMAGEN_MODEL", "imagen-4.0-generate-001"),
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
            if not result.generated_images:
                raise RuntimeError(str(primary_err))
            image = result.generated_images[0].image
            if hasattr(image, "save"):
                image.save(str(out_path))
            elif getattr(image, "image_bytes", None):
                out_path.write_bytes(image.image_bytes)
            else:
                raise RuntimeError(str(primary_err))
            print(f"Saved image: {out_path}")
            return out_path
        except Exception as secondary_err:
            raise RuntimeError(
                "Image generation failed. Enable billing / image quota on your Google key. "
                f"Details: {primary_err} | {secondary_err}"
            ) from secondary_err


def _image_source(image_path: Path | None):
    if not image_path:
        return None
    from google.genai import types

    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return types.Image(image_bytes=data, mime_type=mime)


def generate_one_clip(
    client,
    prompt: str,
    out_path: Path,
    *,
    image_path: Path | None = None,
    duration_sec: int = 8,
) -> Path:
    from google.genai import types

    model = os.getenv("VIDEO_MODEL", "veo-3.1-generate-preview")
    prompt = with_audio_direction(prompt)
    print(f"Generating clip WITH SOUND ({duration_sec}s) via {model}...")
    print(f"  Prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}")

    source_kwargs: dict = {"prompt": prompt}
    img = _image_source(image_path)
    if img is not None:
        source_kwargs["image"] = img

    config_kwargs = {
        "number_of_videos": 1,
        "duration_seconds": duration_sec,
    }
    # Prefer native audio when the API accepts it
    for audio_flag in (True, None):
        try:
            cfg = dict(config_kwargs)
            if audio_flag is True:
                cfg["generate_audio"] = True
            operation = client.models.generate_videos(
                model=model,
                source=types.GenerateVideosSource(**source_kwargs),
                config=types.GenerateVideosConfig(**cfg),
            )
            break
        except TypeError:
            continue
        except Exception as exc:
            msg = str(exc).lower()
            if audio_flag is True and "generate_audio" in msg:
                print("  Note: generate_audio flag not supported; relying on Veo native audio via prompt.")
                continue
            # retry without audio flag once
            if audio_flag is True:
                continue
            raise
    else:
        # Ultimate fallback older signature
        kwargs = {"model": model, "prompt": prompt}
        if img is not None:
            kwargs["image"] = img
        operation = client.models.generate_videos(**kwargs)

    while not getattr(operation, "done", False):
        print("  Waiting for Google (video + sound)...")
        time.sleep(20)
        operation = client.operations.get(operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"Video operation error: {operation.error}")

    response = getattr(operation, "response", None) or getattr(operation, "result", None)
    if response is None:
        raise RuntimeError(f"Video operation finished with no response: {operation}")

    videos = getattr(response, "generated_videos", None) or []
    if not videos:
        raise RuntimeError(f"No generated videos in response: {response}")

    video = videos[0].video
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(client.files, "download") and getattr(video, "uri", None):
        try:
            client.files.download(file=video)
        except Exception:
            pass
    if hasattr(video, "save"):
        video.save(str(out_path))
    elif getattr(video, "video_bytes", None):
        out_path.write_bytes(video.video_bytes)
    elif getattr(video, "uri", None):
        raise RuntimeError(
            f"Clip ready at URI {video.uri} but could not save locally. "
            "Upgrade google-genai or download the URI manually."
        )
    else:
        raise RuntimeError("Unexpected video payload from API.")

    print(f"  Saved clip: {out_path}")
    return out_path


def stitch_clips(clips: list[Path], out_path: Path) -> Path:
    if len(clips) == 1:
        shutil.copy(clips[0], out_path)
        return out_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "ffmpeg not found — individual clips are in output/.../clips/. "
            "Install ffmpeg to merge into one file with continuous audio.",
            file=sys.stderr,
        )
        return clips[0]

    list_file = out_path.parent / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out_path),
    ]
    print("Stitching video + audio with ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"Final video: {out_path}")
    return out_path


def cmd_image(args: argparse.Namespace) -> None:
    client = get_client(load_api_key())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else OUTPUT / f"image-{stamp}.png"
    generate_image(client, args.prompt, out)


def cmd_video(args: argparse.Namespace) -> None:
    client = get_client(load_api_key())
    target = int(args.duration) if args.duration is not None else int(os.getenv("TARGET_DURATION_SEC", "120"))
    clip_sec = int(args.clip_duration) if args.clip_duration is not None else int(os.getenv("CLIP_DURATION_SEC", "8"))
    # Allow env defaults only when CLI uses argparse defaults — prefer explicit CLI
    if os.getenv("TARGET_DURATION_SEC") and args.duration == 120:
        # keep CLI default unless user set env and didn't pass a custom duration intent
        pass
    target = int(args.duration)
    clip_sec = int(args.clip_duration)
    clip_sec = max(4, min(clip_sec, 8))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = OUTPUT / f"run-{stamp}"
    clips_dir = work / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image) if args.image else None
    if args.make_image and not image_path:
        image_path = work / "reference.png"
        generate_image(client, args.prompt, image_path)

    print(f"Planning {target}s storyboard (clips of {clip_sec}s, with sound)...")
    scenes = plan_scenes(client, args.prompt, target, clip_sec)
    (work / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print(f"Planned {len(scenes)} scenes.")

    clips: list[Path] = []
    for i, scene in enumerate(scenes, start=1):
        clip_path = clips_dir / f"scene-{i:02d}.mp4"
        try:
            generate_one_clip(
                client,
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
    p = argparse.ArgumentParser(
        description="Curveo — create images & videos with sound from a prompt",
    )
    sub = p.add_subparsers(dest="command", required=True)

    img = sub.add_parser("image", help="Generate an image from a prompt")
    img.add_argument("prompt", help="Image prompt")
    img.add_argument("--out", help="Output PNG path")
    img.set_defaults(func=cmd_image)

    vid = sub.add_parser(
        "video",
        help="Generate a long video with sound (plans + Veo clips + stitch)",
    )
    vid.add_argument("prompt", help="Video prompt / creative brief")
    vid.add_argument("--image", help="Optional reference image (image-to-video)")
    vid.add_argument(
        "--make-image",
        action="store_true",
        help="Generate a reference image first from the same prompt",
    )
    vid.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Target length in seconds (default: 120)",
    )
    vid.add_argument(
        "--clip-duration",
        type=int,
        default=8,
        help="Seconds per Veo clip (default: 8)",
    )
    vid.add_argument("--out", help="Final MP4 path")
    vid.set_defaults(func=cmd_video)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
