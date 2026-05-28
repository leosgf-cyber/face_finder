import json
import sys
import urllib.request
from pathlib import Path

import cv2
import dlib
import numpy as np

import pickle

MODELS_DIR = Path(__file__).resolve().parent / "models"
SHAPE_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

SHAPE_PREDICTOR_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
FACE_REC_MODEL_URL = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"


def _ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_path, url in [(SHAPE_PREDICTOR, SHAPE_PREDICTOR_URL), (FACE_REC_MODEL, FACE_REC_MODEL_URL)]:
        if model_path.exists():
            continue

        bz2_path = model_path.with_suffix(".dat.bz2")
        print(f"Baixando modelo: {model_path.name}...")
        urllib.request.urlretrieve(url, str(bz2_path))

        import bz2
        with open(bz2_path, "rb") as f_in:
            data = bz2.decompress(f_in.read())
        with open(model_path, "wb") as f_out:
            f_out.write(data)
        bz2_path.unlink()
        print(f"  Modelo salvo em '{model_path}'")


def _get_detector_and_encoder():
    _ensure_models()
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR))
    face_encoder = dlib.face_recognition_model_v1(str(FACE_REC_MODEL))
    return detector, shape_predictor, face_encoder


# Worker-process globals — populated by _worker_init() exactly once per worker.
# Process boundary ensures no contention with the main process or other workers.
_WORKER_DETECTOR = None
_WORKER_SHAPE_PREDICTOR = None
_WORKER_FACE_ENCODER = None


def _worker_init():
    """Initializer for ProcessPoolExecutor workers.

    Loads dlib models once per worker process into module globals so subsequent
    frame processing reuses them. Each worker pays ~100MB resident model cost.
    """
    global _WORKER_DETECTOR, _WORKER_SHAPE_PREDICTOR, _WORKER_FACE_ENCODER
    _ensure_models()
    _WORKER_DETECTOR = dlib.get_frontal_face_detector()
    _WORKER_SHAPE_PREDICTOR = dlib.shape_predictor(str(SHAPE_PREDICTOR))
    _WORKER_FACE_ENCODER = dlib.face_recognition_model_v1(str(FACE_REC_MODEL))


def _worker_process_frame(frame_index: int, frame_path: str):
    """Read a frame, detect+encode faces, return (idx, path, [(encoding, jpeg_bytes)]).

    Pure CPU-bound. Called from worker processes. The main process compares
    encodings against the known references — workers don't know about references.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return frame_index, frame_path, []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _WORKER_DETECTOR(small_rgb, 1)

    faces = []
    for face in detected:
        if scale != 1.0:
            orig_face = dlib.rectangle(
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )
            shape = _WORKER_SHAPE_PREDICTOR(rgb, orig_face)
            encoding = np.array(
                _WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape)
            )
            crop = _crop_face(img, orig_face, padding=0.5)
        else:
            shape = _WORKER_SHAPE_PREDICTOR(rgb, face)
            encoding = np.array(
                _WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape)
            )
            crop = _crop_face(img, face, padding=0.5)

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            continue
        faces.append((encoding, buf.tobytes()))

    return frame_index, frame_path, faces


def _resize_for_detection(rgb, max_width=1200):
    """Resize image for faster face detection, keeping aspect ratio.

    Returns (resized_rgb, scale_factor). scale_factor is 1.0 if no resize needed.
    """
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb, 1.0
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _encode_faces(img_rgb, detector, shape_predictor, face_encoder):
    small_rgb, scale = _resize_for_detection(img_rgb)
    faces = detector(small_rgb, 1)
    encodings = []
    for face in faces:
        if scale != 1.0:
            # Map detection back to original image for better encoding
            orig_face = dlib.rectangle(
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )
            shape = shape_predictor(img_rgb, orig_face)
            encoding = np.array(face_encoder.compute_face_descriptor(img_rgb, shape))
        else:
            shape = shape_predictor(img_rgb, face)
            encoding = np.array(face_encoder.compute_face_descriptor(img_rgb, shape))
        encodings.append(encoding)
    return encodings


def _get_ref_files(ref_dir: Path, extensions: set) -> list:
    """Collect all reference image files, preserving order."""
    ref_files = []
    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
    uses_folders = len(subdirs) > 0

    if uses_folders:
        for person_dir in sorted(subdirs):
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    ref_files.append(img_path)
    else:
        for img_path in sorted(ref_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                ref_files.append(img_path)
    return ref_files


def _cache_is_valid(cache_path: Path, ref_files: list) -> bool:
    """Check if cache exists and is newer than all reference files."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for f in ref_files:
        if f.stat().st_mtime > cache_mtime:
            return False
    return True


def load_references(references_dir: str, tolerance: float = 0.6) -> dict:
    ref_dir = Path(references_dir)
    if not ref_dir.exists():
        print(f"Erro: pasta de referências não encontrada: {references_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    cache_path = ref_dir / ".encodings_cache.pkl"

    # Collect all reference files to check cache validity
    ref_files = _get_ref_files(ref_dir, extensions)

    if _cache_is_valid(cache_path, ref_files):
        print("Carregando referências do cache...")
        with open(cache_path, "rb") as f:
            people = pickle.load(f)

        print(f"Referências carregadas (cache): {len(people)} pessoa(s)")
        for name, encs in people.items():
            print(f"  - {name}: {len(encs)} foto(s)")
        return people

    detector, shape_predictor, face_encoder = _get_detector_and_encoder()

    people = {}

    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
    uses_folders = len(subdirs) > 0

    if uses_folders:
        for person_dir in sorted(subdirs):
            name = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in extensions:
                    continue
                _load_face(img_path, name, people, detector, shape_predictor, face_encoder)
    else:
        for img_path in sorted(ref_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            name = img_path.stem.rsplit("_", 1)[0]
            _load_face(img_path, name, people, detector, shape_predictor, face_encoder)

    if not people:
        print("Erro: nenhum rosto de referência foi carregado.")
        sys.exit(1)

    # Save cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(people, f)
        print("Cache de referências salvo.")
    except Exception as e:
        print(f"Aviso: não foi possível salvar cache: {e}")

    print(f"Referências carregadas: {len(people)} pessoa(s)")
    for name, encs in people.items():
        print(f"  - {name}: {len(encs)} foto(s)")

    return people


def _load_face(img_path, name, people, detector, shape_predictor, face_encoder):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Aviso: não foi possível ler '{img_path.name}', pulando.")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # _encode_faces already uses _resize_for_detection internally
    encodings = _encode_faces(rgb, detector, shape_predictor, face_encoder)

    if not encodings:
        print(f"  Aviso: nenhum rosto encontrado em '{img_path.name}', pulando.")
        return

    if name not in people:
        people[name] = []
    people[name].append(encodings[0])


def _frames_are_similar(frame_a, frame_b, threshold=0.95):
    if frame_a is None or frame_b is None:
        return False
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    small_a = cv2.resize(gray_a, (160, 90))
    small_b = cv2.resize(gray_b, (160, 90))
    score = np.mean(np.abs(small_a.astype(float) - small_b.astype(float)))
    return score < (1 - threshold) * 255


def _detect_and_encode(args):
    frame_path, detector_data = args
    detector, shape_predictor, face_encoder = detector_data

    img = cv2.imread(str(frame_path))
    if img is None:
        return frame_path, img, []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    faces = detector(small_rgb, 1)
    face_encodings = []
    for face in faces:
        if scale != 1.0:
            orig_face = dlib.rectangle(
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )
            shape = shape_predictor(rgb, orig_face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            face_encodings.append((encoding, orig_face))
        else:
            shape = shape_predictor(rgb, face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            face_encodings.append((encoding, face))

    return frame_path, img, face_encodings


def scan_frames(
    frames_dir: str,
    references: dict,
    tolerance: float = 0.6,
    fps: float = 1.0,
    matches_dir: str | None = None,
) -> dict:
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.jpg"))

    if not frame_files:
        print(f"Nenhum frame encontrado em '{frames_dir}'")
        return {}

    if matches_dir:
        Path(matches_dir).mkdir(parents=True, exist_ok=True)

    detector, shape_predictor, face_encoder = _get_detector_and_encoder()

    all_known_encodings = []
    all_known_names = []
    for name, encodings in references.items():
        for enc in encodings:
            all_known_encodings.append(enc)
            all_known_names.append(name)

    all_known_encodings = np.array(all_known_encodings)

    prev_frame = None
    skipped = 0
    to_process = []
    for i, frame_file in enumerate(frame_files):
        img = cv2.imread(str(frame_file))
        if img is None:
            continue
        if _frames_are_similar(prev_frame, img):
            skipped += 1
            prev_frame = img
            continue
        prev_frame = img
        to_process.append((i, frame_file))

    print(f"\nVarrendo {len(to_process)} frames ({skipped} similares pulados de {len(frame_files)} total)...")

    results = {}
    for name in references:
        results[name] = []

    match_counter = 0

    for idx, (i, frame_file) in enumerate(to_process):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processando {idx + 1}/{len(to_process)}...")

        img = cv2.imread(str(frame_file))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_rgb, scale = _resize_for_detection(rgb)
        faces = detector(small_rgb, 1)
        face_encodings = []
        for face in faces:
            if scale != 1.0:
                orig_face = dlib.rectangle(
                    int(face.left() / scale),
                    int(face.top() / scale),
                    int(face.right() / scale),
                    int(face.bottom() / scale),
                )
                shape = shape_predictor(rgb, orig_face)
                encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                face_encodings.append((encoding, orig_face))
            else:
                shape = shape_predictor(rgb, face)
                encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                face_encodings.append((encoding, face))

        for face_enc, face_rect in face_encodings:
            distances = np.linalg.norm(all_known_encodings - face_enc, axis=1)
            best_idx = np.argmin(distances)

            if distances[best_idx] <= tolerance:
                matched_name = all_known_names[best_idx]
                frame_number = i + 1
                timestamp_seconds = frame_number / fps
                timestamp = format_timestamp(timestamp_seconds)

                entry = {
                    "frame": frame_file.name,
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "confidence": round(1 - float(distances[best_idx]), 3),
                }

                if matches_dir:
                    match_img = _crop_face(img, face_rect, padding=0.5)
                    match_filename = f"match_{match_counter:04d}.jpg"
                    cv2.imwrite(str(Path(matches_dir) / match_filename), match_img)
                    entry["match_image"] = match_filename
                    match_counter += 1

                if entry not in results[matched_name]:
                    results[matched_name].append(entry)

    results = {name: matches for name, matches in results.items() if matches}
    return results


def _crop_face(img, face_rect, padding=0.5):
    h, w = img.shape[:2]
    top, bottom = face_rect.top(), face_rect.bottom()
    left, right = face_rect.left(), face_rect.right()
    face_h = bottom - top
    face_w = right - left
    pad_h = int(face_h * padding)
    pad_w = int(face_w * padding)
    top = max(0, top - pad_h)
    bottom = min(h, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(w, right + pad_w)
    return img[top:bottom, left:right]


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_results(results: dict):
    if not results:
        print("\nNenhuma pessoa identificada nos frames.")
        return

    print("\n" + "=" * 50)
    print("RESULTADOS DA VARREDURA")
    print("=" * 50)

    for name, matches in results.items():
        print(f"\n{name}: encontrado(a) em {len(matches)} frame(s)")
        for m in matches:
            print(f"  - {m['timestamp']} | {m['frame']} | confiança: {m['confidence']}")


def save_results(results: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em '{output_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Varrer frames buscando rostos conhecidos")
    parser.add_argument("frames", help="Pasta com os frames extraídos")
    parser.add_argument("references", help="Pasta com fotos de referência")
    parser.add_argument("-t", "--tolerance", type=float, default=0.6, help="Tolerância (0-1, menor = mais rígido, default: 0.6)")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS usado na extração (para calcular timestamps)")
    parser.add_argument("-o", "--output", default="results.json", help="Arquivo de saída JSON (default: results.json)")

    args = parser.parse_args()
    refs = load_references(args.references, args.tolerance)
    results = scan_frames(args.frames, refs, args.tolerance, args.fps)
    print_results(results)
    save_results(results, args.output)
