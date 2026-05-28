#!/usr/bin/env python3
"""
Face Finder — Identificação de pessoas em vídeos.

Pipeline:
  1. Extrai frames de um vídeo (com timestamps configuráveis)
  2. Varre os frames comparando com fotos de referência
  3. Gera relatório em texto e JSON

Uso:
  python main.py pipeline <video> <referencias> [opções]
  python main.py extract <video> [opções]
  python main.py scan <frames> <referencias> [opções]
"""

import argparse
import sys
from pathlib import Path

from extractor import extract_frames
from scanner import load_references, scan_frames, print_results, save_results


def cmd_extract(args):
    extract_frames(args.video, args.output, args.fps, args.start, args.end)


def cmd_scan(args):
    refs = load_references(args.references, args.tolerance)
    results = scan_frames(args.frames, refs, args.tolerance, args.fps)
    print_results(results)
    save_results(results, args.output)


def cmd_pipeline(args):
    frames_dir = args.frames_output or f"./frames_{Path(args.video).stem}"

    print("=" * 50)
    print("ETAPA 1: EXTRAÇÃO DE FRAMES")
    print("=" * 50)
    extract_frames(args.video, frames_dir, args.fps, args.start, args.end)

    print("\n" + "=" * 50)
    print("ETAPA 2: VARREDURA FACIAL")
    print("=" * 50)
    refs = load_references(args.references, args.tolerance)
    results = scan_frames(frames_dir, refs, args.tolerance, args.fps)
    print_results(results)
    save_results(results, args.json_output)


def main():
    parser = argparse.ArgumentParser(
        description="Face Finder — Identificação de pessoas em vídeos",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- pipeline ---
    p_pipe = subparsers.add_parser("pipeline", help="Executar pipeline completo (extrair + varrer)")
    p_pipe.add_argument("video", help="Caminho do vídeo")
    p_pipe.add_argument("references", help="Pasta com fotos de referência")
    p_pipe.add_argument("--fps", type=float, default=1.0, help="Frames por segundo (default: 1.0)")
    p_pipe.add_argument("--start", help="Timestamp inicial (ex: 01:33)")
    p_pipe.add_argument("--end", help="Timestamp final (ex: 22:21)")
    p_pipe.add_argument("-t", "--tolerance", type=float, default=0.6, help="Tolerância facial (0-1, default: 0.6)")
    p_pipe.add_argument("--frames-output", help="Pasta para frames (default: ./frames_<nome_video>)")
    p_pipe.add_argument("-o", "--json-output", default="results.json", help="Saída JSON (default: results.json)")
    p_pipe.set_defaults(func=cmd_pipeline)

    # --- extract ---
    p_ext = subparsers.add_parser("extract", help="Apenas extrair frames do vídeo")
    p_ext.add_argument("video", help="Caminho do vídeo")
    p_ext.add_argument("-o", "--output", default="./frames", help="Pasta de saída (default: ./frames)")
    p_ext.add_argument("--fps", type=float, default=1.0, help="Frames por segundo (default: 1.0)")
    p_ext.add_argument("--start", help="Timestamp inicial")
    p_ext.add_argument("--end", help="Timestamp final")
    p_ext.set_defaults(func=cmd_extract)

    # --- scan ---
    p_scan = subparsers.add_parser("scan", help="Apenas varrer frames existentes")
    p_scan.add_argument("frames", help="Pasta com frames")
    p_scan.add_argument("references", help="Pasta com fotos de referência")
    p_scan.add_argument("-t", "--tolerance", type=float, default=0.6, help="Tolerância facial (0-1, default: 0.6)")
    p_scan.add_argument("--fps", type=float, default=1.0, help="FPS usado na extração")
    p_scan.add_argument("-o", "--output", default="results.json", help="Saída JSON (default: results.json)")
    p_scan.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
