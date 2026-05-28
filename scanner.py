import json
import os
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing

import cv2
import dlib
import numpy as np

import pickle

MODELS_DIR = Path(__file__).resolve().parent / "models"
SHAPE_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

SHAPE_PREDICTOR_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
FACE_REC_MODEL_URL = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"

# Below this many to-process frames, skip the pool — spawn cost would dominate.
PARALLEL_THRESHOLD = 30

# Below this many uncached reference images, skip the pool for ref loading.
REFERENCE_PARALLEL_THRESHOLD = 10


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


# Low-light detection enhancement: skip when mean L (LAB) ≥ this threshold.
# Tuned empirically — well-lit wedding footage typically sits at 120-180.
_ENHANCE_LUMINANCE_THRESHOLD = 80

# Per-person tolerance softening: a person with more reference shots (sunglasses,
# side angles, low light) gets a slightly looser match threshold. Bonus scales
# linearly with extra references beyond TOLERANCE_BONUS_FREE_REFS, capped at MAX.
TOLERANCE_BONUS_FREE_REFS = 2
TOLERANCE_BONUS_PER_REF = 0.025
TOLERANCE_BONUS_MAX = 0.08


def _compute_per_person_tolerance(references, base_tolerance):
    """Return {name: effective_tolerance} based on how many references each person has.

    People with 1-2 reference images use base_tolerance unchanged.
    Each additional reference adds TOLERANCE_BONUS_PER_REF, capped at TOLERANCE_BONUS_MAX.
    """
    out = {}
    for name, encodings in references.items():
        extra = max(0, len(encodings) - TOLERANCE_BONUS_FREE_REFS)
        bonus = min(extra * TOLERANCE_BONUS_PER_REF, TOLERANCE_BONUS_MAX)
        out[name] = base_tolerance + bonus
    return out


def _enhance_for_detection(rgb, luminance_threshold=_ENHANCE_LUMINANCE_THRESHOLD):
    """Brighten dark images for face detection. Returns enhanced rgb or input.

    Adaptive: when the image is already well-lit (mean L >= threshold), the
    input is returned unchanged with no extra cost. Otherwise applies CLAHE on
    the L channel + gamma 1.5 to recover faces in shadow.

    The enhanced image is intended for DETECTION ONLY. Encoding should run
    against the original rgb so embeddings stay comparable with reference
    encodings made under different lighting.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    if float(l_channel.mean()) >= luminance_threshold:
        return rgb
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.5
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(enhanced, table)


def _detect_faces(rgb, detector):
    """Run face detection with low-light enhancement + brightening fallback.

    1. First attempt: detect on the (adaptively) enhanced image.
    2. If zero faces found, retry on a brightened copy of the original.

    Returns the detector's rectangles (same type as `detector(rgb, 1)`),
    in the coordinate space of the input `rgb`.
    """
    enhanced = _enhance_for_detection(rgb)
    detected = detector(enhanced, 1)
    if len(detected) > 0:
        return detected
    bright = cv2.convertScaleAbs(rgb, alpha=1.5, beta=30)
    return detector(bright, 1)


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
    detected = _detect_faces(small_rgb, _WORKER_DETECTOR)

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


def _worker_process_reference(args):
    """Process a single reference image. Worker-side helper.

    args: (img_path_str, name)
    returns: (name, encoding_or_None, img_path_str)
    """
    img_path_str, name = args
    img = cv2.imread(img_path_str)
    if img is None:
        return name, None, img_path_str

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _detect_faces(small_rgb, _WORKER_DETECTOR)
    if not detected:
        return name, None, img_path_str

    face = detected[0]
    if scale != 1.0:
        orig_face = dlib.rectangle(
            int(face.left() / scale),
            int(face.top() / scale),
            int(face.right() / scale),
            int(face.bottom() / scale),
        )
        shape = _WORKER_SHAPE_PREDICTOR(rgb, orig_face)
    else:
        shape = _WORKER_SHAPE_PREDICTOR(rgb, face)
    encoding = np.array(_WORKER_FACE_ENCODER.compute_face_descriptor(rgb, shape))
    return name, encoding, img_path_str


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
    faces = _detect_faces(small_rgb, detector)
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
    ref_files = _get_ref_files(ref_dir, extensions)

    if _cache_is_valid(cache_path, ref_files):
        print("Carregando referências do cache...")
        with open(cache_path, "rb") as f:
            people = pickle.load(f)
        print(f"Referências carregadas (cache): {len(people)} pessoa(s)")
        for name, encs in people.items():
            print(f"  - {name}: {len(encs)} foto(s)")
        return people

    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
    uses_folders = len(subdirs) > 0

    tasks = []
    if uses_folders:
        for person_dir in sorted(subdirs):
            name = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    tasks.append((str(img_path), name))
    else:
        for img_path in sorted(ref_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                name = img_path.stem.rsplit("_", 1)[0]
                tasks.append((str(img_path), name))

    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(tasks) >= REFERENCE_PARALLEL_THRESHOLD

    people = {}

    if use_pool:
        max_workers = int(
            os.environ.get(
                "FACE_FINDER_WORKERS", min((os.cpu_count() or 2) - 1, 8)
            )
        )
        max_workers = max(1, max_workers)

        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as executor:
                ordered_results = list(executor.map(_worker_process_reference, tasks))
        except Exception as e:
            print(f"  Aviso: pool falhou ({e}); caindo pra single-process")
            use_pool = False
        else:
            for name, encoding, img_path_str in ordered_results:
                if encoding is None:
                    print(
                        f"  Aviso: nenhum rosto encontrado em "
                        f"'{Path(img_path_str).name}', pulando."
                    )
                    continue
                people.setdefault(name, []).append(encoding)

    if not use_pool:
        detector, shape_predictor, face_encoder = _get_detector_and_encoder()
        for img_path_str, name in tasks:
            img_path = Path(img_path_str)
            _load_face(
                img_path, name, people, detector, shape_predictor, face_encoder
            )

    if not people:
        print("Erro: nenhum rosto de referência foi carregado.")
        sys.exit(1)

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


def scan_frames(
    frames_dir: str,
    references: dict,
    tolerance: float = 0.6,
    fps: float = 1.0,
    matches_dir: str | None = None,
    progress_callback=None,
) -> dict:
    """Scan a directory of extracted frames for known faces.

    Returns: {name: [match_entry, ...]} keyed by reference person name.

    progress_callback: optional callable invoked from the main thread with
        {"frames_done": int, "frames_total": int, "new_matches": [entry, ...]}
        whenever a frame completes that produced matches, plus every 10 frames
        otherwise, plus on the final frame.
    """
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.jpg"))

    if not frame_files:
        print(f"Nenhum frame encontrado em '{frames_dir}'")
        return {}

    if matches_dir:
        Path(matches_dir).mkdir(parents=True, exist_ok=True)

    all_known_encodings = []
    all_known_names = []
    for name, encodings in references.items():
        for enc in encodings:
            all_known_encodings.append(enc)
            all_known_names.append(name)
    all_known_encodings = np.array(all_known_encodings)

    per_person_tolerance = _compute_per_person_tolerance(references, tolerance)

    # Sequential dedup pass — cheap, runs on main.
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

    print(
        f"\nVarrendo {len(to_process)} frames "
        f"({skipped} similares pulados de {len(frame_files)} total)..."
    )

    results = {name: [] for name in references}
    # Continue match numbering across calls that share the same matches_dir
    # (e.g. multi-video jobs, or videos followed by photos in scan_photo).
    initial_match_count = 0
    if matches_dir:
        initial_match_count = len(list(Path(matches_dir).glob("match_*.jpg")))
    match_counter = [initial_match_count]

    def _consume_face_result(frame_index, frame_file_path, faces, img_for_crop=None):
        """Compare each detected face against references, record matches.

        Returns the list of match entries discovered in this frame (for the callback).
        img_for_crop is provided only by the single-process fast path; the parallel
        path passes JPEG bytes inline so we never need to re-read the file.
        """
        new_matches = []
        for face_data in faces:
            if isinstance(face_data, tuple) and len(face_data) == 2:
                first, second = face_data
                if isinstance(second, (bytes, bytearray)):
                    # Parallel path: (encoding_ndarray, jpeg_bytes)
                    encoding = first
                    crop_bytes = bytes(second)
                    face_rect_for_seq = None
                else:
                    # Single-process path: (encoding, face_rect)
                    encoding = first
                    face_rect_for_seq = second
                    crop_bytes = None
            else:
                continue

            distances = np.linalg.norm(all_known_encodings - encoding, axis=1)
            best_idx = int(np.argmin(distances))
            matched_name = all_known_names[best_idx]
            if distances[best_idx] > per_person_tolerance[matched_name]:
                continue

            frame_number = frame_index + 1
            timestamp_seconds = frame_number / fps

            entry = {
                "frame": Path(frame_file_path).name,
                "frame_number": frame_number,
                "timestamp": format_timestamp(timestamp_seconds),
                "confidence": round(1 - float(distances[best_idx]), 3),
            }

            if matches_dir:
                match_filename = f"match_{match_counter[0]:04d}.jpg"
                match_path = Path(matches_dir) / match_filename
                if crop_bytes is not None:
                    match_path.write_bytes(crop_bytes)
                else:
                    crop = _crop_face(img_for_crop, face_rect_for_seq, padding=0.5)
                    cv2.imwrite(str(match_path), crop)
                entry["match_image"] = match_filename
                match_counter[0] += 1

            if entry not in results[matched_name]:
                results[matched_name].append(entry)
                new_matches.append(entry)
        return new_matches

    parallel_enabled = os.environ.get("FACE_FINDER_PARALLEL", "1") != "0"
    use_pool = parallel_enabled and len(to_process) >= PARALLEL_THRESHOLD

    if use_pool:
        max_workers = int(
            os.environ.get(
                "FACE_FINDER_WORKERS", min((os.cpu_count() or 2) - 1, 8)
            )
        )
        max_workers = max(1, max_workers)

        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as executor:
                futures = {
                    executor.submit(_worker_process_frame, i, str(frame_file)): i
                    for i, frame_file in to_process
                }
                frames_done = 0
                frames_since_last_callback = 0
                for future in as_completed(futures):
                    frames_done += 1
                    frames_since_last_callback += 1
                    new_matches = []
                    try:
                        frame_index, frame_file_path, faces = future.result()
                    except Exception as e:
                        print(f"  Aviso: worker falhou em frame: {e}")
                    else:
                        new_matches = _consume_face_result(
                            frame_index, frame_file_path, faces
                        )

                    should_emit = (
                        progress_callback is not None
                        and (
                            bool(new_matches)
                            or frames_since_last_callback >= 10
                            or frames_done == len(to_process)
                        )
                    )
                    if should_emit:
                        progress_callback(
                            {
                                "frames_done": frames_done,
                                "frames_total": len(to_process),
                                "new_matches": new_matches,
                            }
                        )
                        frames_since_last_callback = 0
        except Exception as e:
            print(f"  Aviso: pool falhou ({e}); caindo pra single-process")
            use_pool = False

    if not use_pool:
        detector, shape_predictor, face_encoder = _get_detector_and_encoder()
        frames_done = 0
        frames_since_last_callback = 0
        for idx, (i, frame_file) in enumerate(to_process):
            frames_done += 1
            frames_since_last_callback += 1
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  Processando {idx + 1}/{len(to_process)}...")

            img = cv2.imread(str(frame_file))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            small_rgb, scale = _resize_for_detection(rgb)
            detected = _detect_faces(small_rgb, detector)
            face_data_list = []
            for face in detected:
                if scale != 1.0:
                    orig_face = dlib.rectangle(
                        int(face.left() / scale),
                        int(face.top() / scale),
                        int(face.right() / scale),
                        int(face.bottom() / scale),
                    )
                    shape = shape_predictor(rgb, orig_face)
                    encoding = np.array(
                        face_encoder.compute_face_descriptor(rgb, shape)
                    )
                    face_data_list.append((encoding, orig_face))
                else:
                    shape = shape_predictor(rgb, face)
                    encoding = np.array(
                        face_encoder.compute_face_descriptor(rgb, shape)
                    )
                    face_data_list.append((encoding, face))

            new_matches = _consume_face_result(
                i, str(frame_file), face_data_list, img_for_crop=img
            )

            should_emit = (
                progress_callback is not None
                and (
                    bool(new_matches)
                    or frames_since_last_callback >= 10
                    or frames_done == len(to_process)
                )
            )
            if should_emit:
                progress_callback(
                    {
                        "frames_done": frames_done,
                        "frames_total": len(to_process),
                        "new_matches": new_matches,
                    }
                )
                frames_since_last_callback = 0

    # Determinism: sort each name's matches by frame_number.
    for name in list(results.keys()):
        results[name].sort(key=lambda m: m["frame_number"])

    results = {name: matches for name, matches in results.items() if matches}
    return results


def scan_photo(
    photo_path: str,
    references: dict,
    tolerance: float = 0.6,
    matches_dir: str | None = None,
    progress_callback=None,
) -> dict:
    """Scan a single photo for known faces. Returns {name: [match_entry, ...]}.

    Match entry shape matches scan_frames; timestamp is None for photos.
    The progress callback (if any) fires exactly once with frames_done=1,
    frames_total=1, and the full list of new matches.
    """
    photo_path_obj = Path(photo_path)
    img = cv2.imread(photo_path)
    if img is None:
        print(f"Não foi possível ler '{photo_path_obj.name}'")
        if progress_callback is not None:
            progress_callback({"frames_done": 1, "frames_total": 1, "new_matches": []})
        return {}

    if matches_dir:
        Path(matches_dir).mkdir(parents=True, exist_ok=True)

    all_known_encodings = []
    all_known_names = []
    for name, encodings in references.items():
        for enc in encodings:
            all_known_encodings.append(enc)
            all_known_names.append(name)
    all_known_encodings = np.array(all_known_encodings)
    per_person_tolerance = _compute_per_person_tolerance(references, tolerance)

    detector, shape_predictor, face_encoder = _get_detector_and_encoder()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _detect_faces(small_rgb, detector)

    # Continue match numbering across calls that share matches_dir.
    match_counter = 0
    if matches_dir:
        match_counter = len(list(Path(matches_dir).glob("match_*.jpg")))

    results = {}
    new_matches = []

    for face in detected:
        if scale != 1.0:
            orig_face = dlib.rectangle(
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )
        else:
            orig_face = face
        shape = shape_predictor(rgb, orig_face)
        encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))

        distances = np.linalg.norm(all_known_encodings - encoding, axis=1)
        best_idx = int(np.argmin(distances))
        matched_name = all_known_names[best_idx]
        if distances[best_idx] > per_person_tolerance[matched_name]:
            continue

        entry = {
            "frame": photo_path_obj.name,
            "frame_number": 0,
            "timestamp": None,
            "confidence": round(1 - float(distances[best_idx]), 3),
        }

        if matches_dir:
            crop = _crop_face(img, orig_face, padding=0.5)
            match_filename = f"match_{match_counter:04d}.jpg"
            cv2.imwrite(str(Path(matches_dir) / match_filename), crop)
            entry["match_image"] = match_filename
            match_counter += 1

        results.setdefault(matched_name, []).append(entry)
        new_matches.append(entry)

    if progress_callback is not None:
        progress_callback({
            "frames_done": 1,
            "frames_total": 1,
            "new_matches": new_matches,
        })

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
