#!/usr/bin/env python3
import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor import extract_frames
from scanner import load_references, scan_frames, _get_detector_and_encoder, _crop_face, _resize_for_detection, _detect_faces

import cv2
import dlib
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
REFERENCES_DIR = UPLOAD_DIR / "references"
VIDEOS_DIR = UPLOAD_DIR / "videos"

for d in [UPLOAD_DIR, RESULTS_DIR, REFERENCES_DIR, VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

jobs = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/version")
def get_version():
    static_dir = Path(__file__).resolve().parent / "static"
    templates_dir = Path(__file__).resolve().parent / "templates"
    latest = 0
    for d in [static_dir, templates_dir]:
        for f in d.rglob("*"):
            if f.is_file():
                latest = max(latest, f.stat().st_mtime)
    return jsonify({"version": int(latest)})


@app.route("/api/people", methods=["GET"])
def list_people():
    people = []
    if REFERENCES_DIR.exists():
        for person_dir in sorted(REFERENCES_DIR.iterdir()):
            if person_dir.is_dir():
                photos = [f.name for f in person_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                if photos:
                    first_photo = str(person_dir / photos[0])
                    people.append({"name": person_dir.name, "photo_count": len(photos), "thumb": photos[0]})
                else:
                    people.append({"name": person_dir.name, "photo_count": 0, "thumb": None})
    return jsonify(people)


@app.route("/api/people/<name>/photo/<filename>")
def get_person_photo(name, filename):
    person_dir = REFERENCES_DIR / name
    if person_dir.exists():
        return send_from_directory(str(person_dir), filename)
    return jsonify({"error": "Não encontrado"}), 404


@app.route("/api/people", methods=["POST"])
def add_person():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nome é obrigatório"}), 400

    person_dir = REFERENCES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "Envie pelo menos uma foto"}), 400

    saved = 0
    for f in files:
        if f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                dest = person_dir / f"{name}_{saved + 1}{ext}"
                f.save(str(dest))
                saved += 1

    return jsonify({"name": name, "photos_saved": saved})


@app.route("/api/people/<name>", methods=["DELETE"])
def delete_person(name):
    person_dir = REFERENCES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)
        return jsonify({"deleted": name})
    return jsonify({"error": "Pessoa não encontrada"}), 404


@app.route("/api/videos", methods=["GET"])
def list_videos():
    vids = []
    if VIDEOS_DIR.exists():
        for f in sorted(VIDEOS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                vids.append({"filename": f.name, "original_name": f.stem, "size_mb": size_mb})
    return jsonify(vids)


@app.route("/api/videos", methods=["POST"])
def upload_videos():
    files = request.files.getlist("videos")
    if not files or not files[0].filename:
        return jsonify({"error": "Envie pelo menos um vídeo"}), 400

    saved = []
    for v in files:
        if v.filename:
            original = Path(v.filename).name
            dest = VIDEOS_DIR / original
            if dest.exists():
                stem = Path(v.filename).stem
                ext = Path(v.filename).suffix
                original = f"{stem}_{uuid.uuid4().hex[:4]}{ext}"
                dest = VIDEOS_DIR / original
            v.save(str(dest))
            saved.append(original)

    return jsonify({"uploaded": saved})


@app.route("/api/videos/<filename>", methods=["DELETE"])
def delete_video(filename):
    video_path = VIDEOS_DIR / filename
    if video_path.exists():
        video_path.unlink()
        return jsonify({"deleted": filename})
    return jsonify({"error": "Vídeo não encontrado"}), 404


pending_clusters = {}
scan_jobs = {}


def _match_clusters_to_existing(clusters):
    detector, shape_predictor, face_encoder = _get_detector_and_encoder()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    known_people = {}
    if REFERENCES_DIR.exists():
        for person_dir in REFERENCES_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in extensions:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                small_rgb, scale = _resize_for_detection(rgb)
                detected = _detect_faces(small_rgb, detector)
                for face_rect in detected:
                    if scale != 1.0:
                        orig_face = dlib.rectangle(
                            int(face_rect.left() / scale),
                            int(face_rect.top() / scale),
                            int(face_rect.right() / scale),
                            int(face_rect.bottom() / scale),
                        )
                        shape = shape_predictor(rgb, orig_face)
                    else:
                        shape = shape_predictor(rgb, face_rect)
                    enc = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                    if person_dir.name not in known_people:
                        known_people[person_dir.name] = []
                    known_people[person_dir.name].append(enc)
                    break

    suggestions = {}
    if known_people:
        all_encs = []
        all_names = []
        for name, encs in known_people.items():
            for enc in encs:
                all_encs.append(enc)
                all_names.append(name)
        all_encs = np.array(all_encs)

        for i, cluster in enumerate(clusters):
            rep_enc = cluster["encodings"][0]
            distances = np.linalg.norm(all_encs - rep_enc, axis=1)
            best_idx = np.argmin(distances)
            if distances[best_idx] <= 0.55:
                suggestions[i] = all_names[best_idx]

    return suggestions


def _cluster_faces(faces, tolerance=0.55, merge_tolerance=0.45):
    clusters = []
    for face in faces:
        best_cluster = None
        best_dist = float("inf")
        for cluster in clusters:
            mean_enc = np.mean(cluster["encodings"], axis=0)
            dist = np.linalg.norm(mean_enc - face["encoding"])
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster
        if best_cluster is not None and best_dist <= tolerance:
            best_cluster["encodings"].append(face["encoding"])
            best_cluster["sources"].append(face["source"])
        else:
            clusters.append({
                "encodings": [face["encoding"]],
                "crop": face["crop"],
                "sources": [face["source"]],
            })

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                mean_i = np.mean(clusters[i]["encodings"], axis=0)
                mean_j = np.mean(clusters[j]["encodings"], axis=0)
                if np.linalg.norm(mean_i - mean_j) <= merge_tolerance:
                    clusters[i]["encodings"].extend(clusters[j]["encodings"])
                    clusters[i]["sources"].extend(clusters[j]["sources"])
                    clusters.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    clusters.sort(key=lambda c: len(c["sources"]), reverse=True)
    return clusters


@app.route("/api/scan-ref-video", methods=["POST"])
def scan_ref_video():
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify({"error": "Envie um vídeo"}), 400

    scan_id = uuid.uuid4().hex[:8]
    video_ext = Path(video.filename).suffix
    video_path = RESULTS_DIR / f"refvid_{scan_id}{video_ext}"
    video.save(str(video_path))

    fps = float(request.form.get("fps", 2.0))
    cluster_tolerance = float(request.form.get("cluster_tolerance", 0.55))
    start = request.form.get("start") or None
    end = request.form.get("end") or None

    scan_jobs[scan_id] = {
        "status": "processing",
        "total": 0,
        "processed": 0,
        "faces_found": 0,
        "phase": "extracting",
        "result": None,
    }

    thread = threading.Thread(
        target=_scan_ref_video_job,
        args=(scan_id, str(video_path), fps, start, end, cluster_tolerance),
        daemon=True,
    )
    thread.start()

    return jsonify({"scan_id": scan_id})


def _scan_ref_video_job(scan_id, video_path, fps, start, end, cluster_tolerance=0.55):
    try:
        frames_dir = str(RESULTS_DIR / f"refframes_{scan_id}")
        scan_jobs[scan_id]["phase"] = "extracting"
        extract_frames(video_path, frames_dir, fps, start, end)

        frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
        scan_jobs[scan_id]["total"] = len(frame_files)
        scan_jobs[scan_id]["phase"] = "detecting"

        if not frame_files:
            scan_jobs[scan_id]["status"] = "error"
            scan_jobs[scan_id]["error"] = "Nenhum frame extraído do vídeo"
            return

        detector, shape_predictor, face_encoder = _get_detector_and_encoder()

        faces = []
        for i, frame_path in enumerate(frame_files):
            scan_jobs[scan_id]["processed"] = i + 1

            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            small_rgb, scale = _resize_for_detection(rgb)
            detected = _detect_faces(small_rgb, detector)
            for face_rect in detected:
                if scale != 1.0:
                    orig_face = dlib.rectangle(
                        int(face_rect.left() / scale),
                        int(face_rect.top() / scale),
                        int(face_rect.right() / scale),
                        int(face_rect.bottom() / scale),
                    )
                    shape = shape_predictor(rgb, orig_face)
                    encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                    crop = _crop_face(img, orig_face, padding=0.4)
                else:
                    shape = shape_predictor(rgb, face_rect)
                    encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                    crop = _crop_face(img, face_rect, padding=0.4)
                faces.append({"encoding": encoding, "crop": crop, "source": frame_path.name})
                scan_jobs[scan_id]["faces_found"] = len(faces)

        if not faces:
            scan_jobs[scan_id]["status"] = "error"
            scan_jobs[scan_id]["error"] = "Nenhum rosto detectado no vídeo"
            return

        clusters = _cluster_faces(faces, tolerance=cluster_tolerance)

        thumbs_dir = RESULTS_DIR / f"scan_{scan_id}"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        scan_upload_dir = RESULTS_DIR / f"scan_upload_{scan_id}"
        scan_upload_dir.mkdir(parents=True, exist_ok=True)

        suggestions = _match_clusters_to_existing(clusters)

        cluster_data = []
        for idx, cluster in enumerate(clusters):
            thumb_name = f"face_{idx}.jpg"
            cv2.imwrite(str(thumbs_dir / thumb_name), cluster["crop"])

            for src in set(cluster["sources"]):
                src_path = Path(frames_dir) / src
                if src_path.exists():
                    shutil.copy2(str(src_path), str(scan_upload_dir / src))

            entry = {
                "id": idx,
                "thumb": thumb_name,
                "photo_count": len(set(cluster["sources"])),
                "sources": list(set(cluster["sources"]))[:3],
            }
            if idx in suggestions:
                entry["suggested_name"] = suggestions[idx]
            cluster_data.append(entry)

        pending_clusters[scan_id] = {
            "upload_dir": str(scan_upload_dir),
            "clusters": clusters,
        }

        scan_jobs[scan_id]["status"] = "done"
        scan_jobs[scan_id]["result"] = cluster_data

    except Exception as e:
        scan_jobs[scan_id]["status"] = "error"
        scan_jobs[scan_id]["error"] = str(e)


PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _detect_faces_in_image(img_path, detector, shape_predictor, face_encoder, source_label):
    """Detect every face in a single image, return list of {encoding, crop, source} dicts."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_rgb, scale = _resize_for_detection(rgb)
    detected = _detect_faces(small_rgb, detector)
    out = []
    for face_rect in detected:
        if scale != 1.0:
            orig_face = dlib.rectangle(
                int(face_rect.left() / scale),
                int(face_rect.top() / scale),
                int(face_rect.right() / scale),
                int(face_rect.bottom() / scale),
            )
            shape = shape_predictor(rgb, orig_face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            crop = _crop_face(img, orig_face, padding=0.4)
        else:
            shape = shape_predictor(rgb, face_rect)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            crop = _crop_face(img, face_rect, padding=0.4)
        out.append({"encoding": encoding, "crop": crop, "source": source_label})
    return out


@app.route("/api/scan-folder", methods=["POST"])
def scan_folder():
    photo_files = [f for f in request.files.getlist("photos") if f.filename]
    video_files = [f for f in request.files.getlist("videos") if f.filename]

    if not photo_files and not video_files:
        return jsonify({"error": "Nenhuma foto ou vídeo recebido"}), 400

    scan_id = uuid.uuid4().hex[:8]
    scan_upload_dir = RESULTS_DIR / f"scan_upload_{scan_id}"
    scan_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_photos = []
    for f in photo_files:
        fname = Path(f.filename).name
        if Path(fname).suffix.lower() in PHOTO_EXTENSIONS:
            dest = scan_upload_dir / fname
            f.save(str(dest))
            saved_photos.append(dest)

    saved_videos = []
    if video_files:
        videos_temp_dir = RESULTS_DIR / f"scan_videos_{scan_id}"
        videos_temp_dir.mkdir(parents=True, exist_ok=True)
        for f in video_files:
            fname = Path(f.filename).name
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                dest = videos_temp_dir / fname
                f.save(str(dest))
                saved_videos.append(dest)

    if not saved_photos and not saved_videos:
        return jsonify({"error": "Nenhum arquivo válido na pasta"}), 400

    scan_jobs[scan_id] = {
        "status": "processing",
        "total": len(saved_photos),
        "processed": 0,
        "faces_found": 0,
        "phase": "extracting" if saved_videos else "detecting",
        "result": None,
    }

    cluster_tolerance = float(request.form.get("cluster_tolerance", 0.55))

    thread = threading.Thread(
        target=_scan_folder_job,
        args=(
            scan_id,
            sorted(saved_photos),
            sorted(saved_videos),
            str(scan_upload_dir),
            cluster_tolerance,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"scan_id": scan_id, "total": len(saved_photos) + len(saved_videos)})


def _scan_folder_job(scan_id, saved_photos, saved_videos, upload_dir, cluster_tolerance=0.55):
    try:
        detector, shape_predictor, face_encoder = _get_detector_and_encoder()

        # Phase 1: extract frames from any videos, copy them into upload_dir with
        # video-index prefix so frames from different videos don't collide on name.
        frame_records = []  # [(frame_path_in_upload_dir, source_label)]
        if saved_videos:
            scan_jobs[scan_id]["phase"] = "extracting"
            scan_jobs[scan_id]["total"] = len(saved_videos)
            video_fps = 2.0
            upload_dir_path = Path(upload_dir)
            for vid_idx, video_path in enumerate(saved_videos):
                scan_jobs[scan_id]["processed"] = vid_idx
                frames_dir = RESULTS_DIR / f"scan_vidframes_{scan_id}_{vid_idx}"
                extract_frames(str(video_path), str(frames_dir), video_fps, None, None)
                for fp in sorted(frames_dir.glob("frame_*.jpg")):
                    label = f"v{vid_idx}_{fp.name}"
                    dest = upload_dir_path / label
                    shutil.copy2(str(fp), str(dest))
                    frame_records.append((dest, label))
                shutil.rmtree(frames_dir, ignore_errors=True)
            # Source video files no longer needed
            shutil.rmtree(RESULTS_DIR / f"scan_videos_{scan_id}", ignore_errors=True)

        # Phase 2: detect faces in every photo and extracted frame
        scan_jobs[scan_id]["phase"] = "detecting"
        scan_jobs[scan_id]["total"] = len(saved_photos) + len(frame_records)
        scan_jobs[scan_id]["processed"] = 0

        faces = []
        processed = 0
        for img_path in saved_photos:
            processed += 1
            scan_jobs[scan_id]["processed"] = processed
            faces.extend(_detect_faces_in_image(
                img_path, detector, shape_predictor, face_encoder, img_path.name
            ))
            scan_jobs[scan_id]["faces_found"] = len(faces)

        for frame_path, label in frame_records:
            processed += 1
            scan_jobs[scan_id]["processed"] = processed
            faces.extend(_detect_faces_in_image(
                frame_path, detector, shape_predictor, face_encoder, label
            ))
            scan_jobs[scan_id]["faces_found"] = len(faces)

        if not faces:
            scan_jobs[scan_id]["status"] = "error"
            scan_jobs[scan_id]["error"] = "Nenhum rosto detectado nas imagens"
            return

        clusters = _cluster_faces(faces, tolerance=cluster_tolerance)

        thumbs_dir = RESULTS_DIR / f"scan_{scan_id}"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        suggestions = _match_clusters_to_existing(clusters)

        cluster_data = []
        for i, cluster in enumerate(clusters):
            thumb_name = f"face_{i}.jpg"
            cv2.imwrite(str(thumbs_dir / thumb_name), cluster["crop"])
            entry = {
                "id": i,
                "thumb": thumb_name,
                "photo_count": len(cluster["sources"]),
                "sources": list(set(cluster["sources"]))[:3],
            }
            if i in suggestions:
                entry["suggested_name"] = suggestions[i]
            cluster_data.append(entry)

        pending_clusters[scan_id] = {
            "upload_dir": upload_dir,
            "clusters": clusters,
        }

        scan_jobs[scan_id]["status"] = "done"
        scan_jobs[scan_id]["result"] = cluster_data

    except Exception as e:
        scan_jobs[scan_id]["status"] = "error"
        scan_jobs[scan_id]["error"] = str(e)


@app.route("/api/scan-status/<scan_id>", methods=["GET"])
def scan_status(scan_id):
    if scan_id not in scan_jobs:
        return jsonify({"error": "Scan não encontrado"}), 404
    return jsonify(scan_jobs[scan_id])


@app.route("/api/scan-thumbs/<scan_id>/<filename>")
def get_scan_thumb(scan_id, filename):
    thumbs_path = RESULTS_DIR / f"scan_{scan_id}"
    if thumbs_path.exists():
        return send_from_directory(str(thumbs_path), filename)
    return jsonify({"error": "Não encontrado"}), 404


@app.route("/api/confirm-people", methods=["POST"])
def confirm_people():
    scan_id = request.json.get("scan_id", "")
    assignments = request.json.get("assignments", [])

    if scan_id not in pending_clusters:
        return jsonify({"error": "Scan não encontrado"}), 404

    scan_data = pending_clusters[scan_id]
    clusters = scan_data["clusters"]
    upload_dir = Path(scan_data["upload_dir"])

    saved_count = 0

    for assignment in assignments:
        cluster_id = assignment.get("id")
        name = assignment.get("name", "").strip()
        if not name or cluster_id is None or cluster_id >= len(clusters):
            continue

        person_dir = REFERENCES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)

        cluster = clusters[cluster_id]
        sources = set(cluster["sources"])

        idx = 0
        for src in sources:
            src_path = upload_dir / src
            if src_path.exists():
                ext = src_path.suffix.lower()
                dest = person_dir / f"{name}_{idx + 1}{ext}"
                shutil.copy2(str(src_path), str(dest))
                idx += 1
                saved_count += 1

    del pending_clusters[scan_id]
    return jsonify({"saved": saved_count})


@app.route("/api/load-videos-folder", methods=["POST"])
def load_videos_folder():
    files = request.files.getlist("videos")
    if not files or not files[0].filename:
        return jsonify({"error": "Nenhum vídeo recebido"}), 400

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    loaded = []

    for v in files:
        if v.filename:
            fname = Path(v.filename).name
            if Path(fname).suffix.lower() in video_exts:
                dest = VIDEOS_DIR / fname
                if not dest.exists():
                    v.save(str(dest))
                loaded.append(fname)

    if not loaded:
        return jsonify({"error": "Nenhum vídeo encontrado na pasta"}), 400

    return jsonify({"loaded": loaded, "count": len(loaded)})


@app.route("/api/process", methods=["POST"])
def start_processing():
    selected = request.json.get("videos", []) if request.is_json else []

    if not selected:
        return jsonify({"error": "Selecione pelo menos um vídeo"}), 400

    people_dirs = [d for d in REFERENCES_DIR.iterdir() if d.is_dir()]
    if not people_dirs:
        return jsonify({"error": "Cadastre pelo menos uma pessoa antes de processar"}), 400

    job_id = str(uuid.uuid4())[:8]

    video_paths = []
    video_names = []
    for fname in selected:
        vpath = VIDEOS_DIR / fname
        if vpath.exists():
            video_paths.append(str(vpath))
            video_names.append(fname)

    req_data = request.json if request.is_json else {}
    fps = float(req_data.get("fps", 1.0))
    tolerance = float(req_data.get("tolerance", 0.6))
    start = req_data.get("start") or None
    end = req_data.get("end") or None

    jobs[job_id] = {
        "status": "processing",
        "progress": "Iniciando...",
        "videos": video_names,
        "results": None,
        "partial_matches": 0,
        "frames_done": 0,
        "frames_total": 0,
        "live_matches": [],
    }

    thread = threading.Thread(
        target=_process_job,
        args=(job_id, video_paths, video_names, fps, tolerance, start, end),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _process_job(job_id, video_paths, video_names, fps, tolerance, start, end):
    try:
        matches_dir = str(RESULTS_DIR / f"matches_{job_id}")
        Path(matches_dir).mkdir(parents=True, exist_ok=True)

        jobs[job_id]["progress"] = "Carregando referências..."
        refs = load_references(str(REFERENCES_DIR), tolerance)

        all_results = {}

        for idx, (video_path, video_name) in enumerate(zip(video_paths, video_names)):
            jobs[job_id]["progress"] = (
                f"Extraindo frames: {video_name} ({idx + 1}/{len(video_paths)})..."
            )
            frames_dir = str(RESULTS_DIR / f"frames_{job_id}_{idx}")
            extract_frames(video_path, frames_dir, fps, start, end)

            jobs[job_id]["progress"] = (
                f"Varrendo: {video_name} ({idx + 1}/{len(video_paths)})..."
            )
            # Reset per-video frame counter; live_matches accumulates across videos.
            jobs[job_id]["frames_done"] = 0
            jobs[job_id]["frames_total"] = 0

            def _on_progress(payload, _video_name=video_name):
                jobs[job_id]["frames_done"] = payload["frames_done"]
                jobs[job_id]["frames_total"] = payload["frames_total"]
                for m in payload["new_matches"]:
                    m_with_video = dict(m)
                    m_with_video["video"] = _video_name
                    jobs[job_id]["live_matches"].append(m_with_video)
                jobs[job_id]["partial_matches"] = len(jobs[job_id]["live_matches"])

            results = scan_frames(
                frames_dir,
                refs,
                tolerance,
                fps,
                matches_dir,
                progress_callback=_on_progress,
            )

            for name, matches in results.items():
                for m in matches:
                    m["video"] = video_name
                if name not in all_results:
                    all_results[name] = []
                all_results[name].extend(matches)

            total_matches = sum(len(v) for v in all_results.values())
            people_found = len(all_results)
            jobs[job_id]["partial_matches"] = total_matches
            jobs[job_id]["progress"] = (
                f"Video {idx + 1}/{len(video_paths)} concluido | "
                f"{people_found} pessoa(s), {total_matches} match(es)"
            )

            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass

        result_path = RESULTS_DIR / f"{job_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        summary = {}
        for name, matches in all_results.items():
            summary[name] = {
                "total_appearances": len(matches),
                "timestamps": [m["timestamp"] for m in matches],
                "videos": list(set(m["video"] for m in matches)),
                "avg_confidence": round(sum(m["confidence"] for m in matches) / len(matches), 3) if matches else 0,
                "best_matches": matches[:6],
            }

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = "Concluído!"
        jobs[job_id]["results"] = summary
        jobs[job_id]["full_results"] = all_results

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = f"Erro: {str(e)}"


@app.route("/api/jobs/<job_id>", methods=["GET"])
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job não encontrado"}), 404
    return jsonify(jobs[job_id])


@app.route("/api/jobs/<job_id>/download", methods=["GET"])
def download_results(job_id):
    result_file = RESULTS_DIR / f"{job_id}.json"
    if result_file.exists():
        return send_from_directory(str(RESULTS_DIR), f"{job_id}.json", as_attachment=True)
    return jsonify({"error": "Resultados não encontrados"}), 404


@app.route("/api/matches/<job_id>/<filename>")
def get_match_image(job_id, filename):
    matches_path = RESULTS_DIR / f"matches_{job_id}"
    if matches_path.exists():
        return send_from_directory(str(matches_path), filename)
    return jsonify({"error": "Não encontrado"}), 404


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    elif path.is_file():
        total = path.stat().st_size
    return total


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    """Delete temporary frames, scan, and refvid files from results/."""
    freed = 0
    deleted_items = []

    if not RESULTS_DIR.exists():
        return jsonify({"freed_bytes": 0, "freed_mb": 0, "deleted": []})

    prefixes_dirs = ("frames_", "scan_upload_", "refframes_", "scan_")
    prefixes_files = ("refvid_",)

    for item in list(RESULTS_DIR.iterdir()):
        if item.is_dir() and any(item.name.startswith(p) for p in prefixes_dirs):
            size = _get_dir_size(item)
            shutil.rmtree(item)
            freed += size
            deleted_items.append(item.name)
        elif item.is_file() and any(item.name.startswith(p) for p in prefixes_files):
            size = item.stat().st_size
            item.unlink()
            freed += size
            deleted_items.append(item.name)

    return jsonify({
        "freed_bytes": freed,
        "freed_mb": round(freed / (1024 * 1024), 2),
        "deleted": deleted_items,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(debug=True, host="0.0.0.0", port=args.port)
