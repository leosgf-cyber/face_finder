import subprocess
import shutil
import sys
from pathlib import Path


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("Erro: ffmpeg não encontrado. Instale com:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def timestamp_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    check_ffmpeg()

    video = Path(video_path)
    if not video.exists():
        print(f"Erro: vídeo não encontrado: {video_path}")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]

    if start:
        cmd += ["-ss", str(timestamp_to_seconds(start))]
    if end:
        duration_flag = []
        if start:
            duration = timestamp_to_seconds(end) - timestamp_to_seconds(start)
            duration_flag = ["-t", str(duration)]
        else:
            duration_flag = ["-to", str(timestamp_to_seconds(end))]
        cmd += duration_flag

    cmd += [
        "-i", str(video),
        "-vf", f"fps={fps},scale=-1:1080:force_original_aspect_ratio=decrease",
        "-frame_pts", "1",
        str(out / "frame_%06d.jpg"),
    ]

    print(f"Extraindo frames de '{video.name}' a {fps} fps...")
    if start or end:
        print(f"  Intervalo: {start or 'início'} → {end or 'fim'}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erro no ffmpeg:\n{result.stderr}")
        sys.exit(1)

    frame_count = len(list(out.glob("frame_*.jpg")))
    print(f"  {frame_count} frames extraídos em '{out}'")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrair frames de um vídeo")
    parser.add_argument("video", help="Caminho do vídeo")
    parser.add_argument("-o", "--output", default="./frames", help="Pasta de saída (default: ./frames)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames por segundo (default: 1.0)")
    parser.add_argument("--start", help="Timestamp inicial (ex: 01:33)")
    parser.add_argument("--end", help="Timestamp final (ex: 22:21)")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps, args.start, args.end)
