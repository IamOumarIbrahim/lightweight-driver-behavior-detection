"""
DMD (Driver Monitoring Dataset) Frame Extraction & Cropping Pipeline
====================================================================

This script provides a reproducible, end-to-end pipeline to:
1. Extract 1 frame per second (1 fps) from all 1280x720 `rgb_face` video streams.
2. Mirror the DMD dataset hierarchy (category -> group -> subject -> session) in raw frames.
3. Apply the standardized 640x640 face region cropping box:
   - Box: x=272, y=71, width=640, height=640 (Coordinates: (272, 71) to (912, 711))
4. Organize cropped images into the canonical benchmark structure:
   data/processed/RGB/images/subject_XX/video_YY/subject_XX_video_YY_frame_ZZZZ.jpg
5. Verify extraction integrity, checking for black/corrupt frames and sharpness.

Usage:
------
# Full pipeline (extraction + cropping + verification)
python scripts/data/extract_and_crop_dmd.py

# Custom parameters example
python -m tools.data.extract_rgb
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# 14 unique subjects in DMD dataset sorted numerically
ORIGINAL_SUBJECTS = [1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37]
SUBJECT_MAP = {
    orig_id: f"subject_{idx:02d}"
    for idx, orig_id in enumerate(ORIGINAL_SUBJECTS, start=1)
}

SESSION_MAP = {
    ("distraction", "s1"): "video_01",
    ("distraction", "s2"): "video_02",
    ("distraction", "s3"): "video_03",
    ("di21-dmd-dataset-drowsiness", "s5"): "video_04",
    ("gaze", "s6"): "video_05",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DMD Dataset Frame Extraction & 640x640 Face Cropping Pipeline"
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_dmd = os.path.join(repo_root, "data", "DMD")
    default_raw_imgs = os.path.join(repo_root, "data", "processed", "RGB", "raw_1fps")
    default_cropped_imgs = os.path.join(repo_root, "data", "processed", "RGB", "images")

    parser.add_argument(
        "--dmd-dir",
        type=str,
        default=default_dmd,
        help="Path to DMD dataset root containing distraction, gaze, drowsiness folders.",
    )
    parser.add_argument(
        "--out-raw-images",
        type=str,
        default=default_raw_imgs,
        help="Output directory for full-resolution (1280x720) 1 fps extracted frames.",
    )
    parser.add_argument(
        "--out-cropped",
        type=str,
        default=default_cropped_imgs,
        help="Output directory for 640x640 cropped face frames.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Frame extraction rate in frames per second (default: 1.0 = 1 frame every 1s).",
    )
    parser.add_argument(
        "--crop-box",
        nargs=4,
        type=int,
        default=[272, 71, 640, 640],
        metavar=("X", "Y", "W", "H"),
        help="Face bounding box crop [x, y, width, height] (default: 272 71 640 640).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 4),
        help="Number of parallel worker processes (default: 6).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG compression quality (1-100, default: 95).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip raw 1 fps extraction step and run only cropping on existing raw images.",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="Skip cropping step and run only raw frame extraction.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run only quality verification (black frames & blur checks) on cropped dataset.",
    )
    return parser.parse_args()


# ==========================================
# STEP 1: Video Discovery & 1 FPS Extraction
# ==========================================


def discover_rgb_face_videos(dmd_dir, raw_out_base):
    categories = ["distraction", "gaze", "di21-dmd-dataset-drowsiness"]
    sources = []

    for cat in categories:
        cat_path = os.path.join(dmd_dir, cat)
        if not os.path.exists(cat_path):
            continue
        for root, _, files in os.walk(cat_path):
            for f in files:
                if f.endswith("_rgb_face.mp4"):
                    full_path = os.path.join(root, f)
                    rel = os.path.relpath(full_path, dmd_dir)
                    parts = rel.split(os.sep)
                    if len(parts) < 5:
                        continue

                    cat_name = parts[0]
                    cat_prefix = (
                        "drowsiness" if "drowsiness" in cat_name.lower() else cat_name
                    )
                    group = parts[1]
                    subject = parts[2]
                    session = parts[3]
                    out_dir = os.path.join(
                        raw_out_base, cat_name, group, subject, session
                    )

                    sources.append(
                        {
                            "video_path": full_path,
                            "out_dir": out_dir,
                            "cat_name": cat_name,
                            "cat_prefix": cat_prefix,
                            "group": group,
                            "subject": subject,
                            "session": session,
                            "filename": f,
                        }
                    )
    # Multiple recordings can map to one canonical subject/session. Preserve
    # the frozen dataset's observed overlay deterministically: sources are
    # applied in ascending path order, so the later recording replaces the
    # shared prefix while a longer earlier recording retains its tail.
    grouped = defaultdict(list)
    for source in sources:
        grouped[source["out_dir"]].append(source)
    return [
        {
            "out_dir": out_dir,
            "sources": sorted(group, key=lambda item: item["video_path"]),
        }
        for out_dir, group in sorted(grouped.items())
    ]


def extract_video_worker(task, sample_fps, jpeg_quality):
    t0 = time.time()
    os.makedirs(task["out_dir"], exist_ok=True)
    cap = cv2.VideoCapture(task["video_path"])
    if not cap.isOpened():
        return task, 0, 0, f"Cannot open video {task['video_path']}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    extracted_count = 0
    next_target_frame = 0

    cat_prefix = task["cat_prefix"]
    group = task["group"]
    subject = task["subject"]
    session = task["session"]
    out_dir = task["out_dir"]

    step_frames = fps / sample_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == next_target_frame:
            extracted_count += 1
            frame_filename = (
                f"{cat_prefix}_{group}_{subject}_{session}_{extracted_count:04d}.jpg"
            )
            out_path = os.path.join(out_dir, frame_filename)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            next_target_frame = round(extracted_count * step_frames)

        frame_idx += 1

    cap.release()
    dur = round(time.time() - t0, 1)
    return task, extracted_count, dur, None


def extract_session_worker(task, sample_fps, jpeg_quality):
    counts = []
    durations = []
    for source in task["sources"]:
        _, count, duration, error = extract_video_worker(
            source, sample_fps, jpeg_quality
        )
        if error:
            return task, counts, durations, error
        counts.append(count)
        durations.append(duration)
    return task, counts, durations, None


def run_extraction(tasks, workers, sample_fps, jpeg_quality):
    source_count = sum(len(task["sources"]) for task in tasks)
    print(
        f"\n[Step 1/3] Extracting frames at {sample_fps} FPS across {source_count} recordings into {len(tasks)} canonical sessions..."
    )
    occupied = [
        task["out_dir"]
        for task in tasks
        if os.path.isdir(task["out_dir"])
        and any(name.lower().endswith(".jpg") for name in os.listdir(task["out_dir"]))
    ]
    if occupied:
        raise RuntimeError(
            f"Refusing extraction into {len(occupied)} non-empty session directories; use a new --out-raw-images directory"
        )
    t_start = time.time()
    total_frames = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_session_worker, t, sample_fps, jpeg_quality): t
            for t in tasks
        }
        for f in as_completed(futures):
            task, counts, durations, err = f.result()
            completed += 1
            retained_count = max(counts, default=0)
            total_frames += retained_count
            pct = round((completed / len(tasks)) * 100, 1)
            if err:
                print(f"  [{pct}%] FAILED {task['out_dir']}: {err}")
            else:
                print(
                    f"  [{pct}%] ({completed}/{len(tasks)}) {task['out_dir']}: {retained_count} frames from {len(counts)} source(s) ({sum(durations):.1f}s)"
                )

    elapsed = round(time.time() - t_start, 1)
    print(
        f"-> Extraction complete: {total_frames} retained frames across {completed} sessions in {elapsed}s.\n"
    )
    return total_frames


# ==========================================
# STEP 2: Face Bounding Box Cropping (640x640)
# ==========================================


def collect_crop_tasks(raw_base, crop_base):
    dir_tasks = []
    manifest_entries = []

    for root, _, files in os.walk(raw_base):
        jpgs = sorted([f for f in files if f.endswith(".jpg")])
        if jpgs:
            rel = os.path.relpath(root, raw_base)
            parts = rel.split(os.sep)
            # parts: [cat, group, sub_id, session]
            if len(parts) >= 4:
                cat, group, sub_id_str, session = parts[0], parts[1], parts[2], parts[3]
                sub_id = int(sub_id_str)
                sub_folder = SUBJECT_MAP.get(sub_id, f"subject_{sub_id:02d}")
                vid_folder = SESSION_MAP.get((cat, session), f"video_{session}")
                out_dir = os.path.join(crop_base, sub_folder, vid_folder)
            else:
                out_dir = os.path.join(crop_base, rel)
                sub_folder, vid_folder = "unknown", "unknown"
                sub_id, cat, session, group = 0, "unknown", "unknown", "unknown"

            dir_tasks.append((root, out_dir, jpgs, sub_folder, vid_folder))
            manifest_entries.append(
                {
                    "subject_folder": sub_folder,
                    "original_subject_id": sub_id,
                    "group": group,
                    "video_folder": vid_folder,
                    "original_category": cat,
                    "original_session": session,
                    "frame_count": len(jpgs),
                    "source_dir": rel,
                    "target_dir": f"{sub_folder}/{vid_folder}",
                    "source_collision_policy": "lexicographic_overlay_later_source_wins_shared_prefix",
                }
            )

    return dir_tasks, manifest_entries


def crop_dir_worker(task, crop_box, jpeg_quality):
    in_dir, out_dir, files, sub_folder, vid_folder = task
    os.makedirs(out_dir, exist_ok=True)
    x, y, w, h = crop_box
    x2, y2 = x + w, y + h
    count = 0
    t0 = time.time()

    for idx, f in enumerate(files, start=1):
        in_path = os.path.join(in_dir, f)
        new_filename = f"{sub_folder}_{vid_folder}_frame_{idx:04d}.jpg"
        out_path = os.path.join(out_dir, new_filename)
        img = cv2.imread(in_path)
        if img is None:
            print(f"  WARNING: Failed to read image {in_path}, skipping")
            continue
        if img.shape[0] < y2 or img.shape[1] < x2:
            print(
                f"  WARNING: unexpected shape {img.shape[:2]} (smaller than crop bounds {y2}x{x2}) for {in_path}, skipping"
            )
            continue
        cropped = img[y:y2, x:x2]
        if cropped.shape[0] != h or cropped.shape[1] != w:
            print(
                f"  WARNING: crop produced {cropped.shape[:2]} instead of ({h}, {w}) for {in_path}, skipping"
            )
            continue
        cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        count += 1

    dur = round(time.time() - t0, 2)
    return in_dir, out_dir, count, dur


def run_cropping(tasks, manifest_entries, crop_base, workers, crop_box, jpeg_quality):
    x, y, w, h = crop_box
    total_imgs = sum(len(t[2]) for t in tasks)
    print(
        f"\n[Step 2/3] Cropping {total_imgs} frames to {w}x{h} (Box: x={x}, y={y}, w={w}, h={h})..."
    )
    t_start = time.time()
    completed_dirs = 0
    total_cropped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(crop_dir_worker, t, crop_box, jpeg_quality): t
            for t in tasks
        }
        for f in as_completed(futures):
            _, out_d, count, dur = f.result()
            completed_dirs += 1
            total_cropped += count
            pct = round((completed_dirs / len(tasks)) * 100, 1)
            print(
                f"  [{pct}%] ({completed_dirs}/{len(tasks)}) Cropped {count} frames ({dur}s) -> {out_d}"
            )

    # Save manifest.json
    manifest_path = os.path.join(os.path.dirname(crop_base), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_entries, mf, indent=2)
    print(f"-> Saved dataset manifest to: {manifest_path}")

    elapsed = round(time.time() - t_start, 1)
    print(
        f"-> Cropping complete: {total_cropped} images across {completed_dirs} directories in {elapsed}s.\n"
    )
    return total_cropped


# ==========================================
# STEP 3: Quality Verification & Blur Check
# ==========================================


def verify_chunk_worker(paths):
    pure_black = 0
    blur_scores = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if np.max(img) == 0:
            pure_black += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(score)
    return pure_black, blur_scores


def verify_cropped_dataset(crop_base, workers):
    print(f"\n[Step 3/3] Running quality verification on cropped dataset: {crop_base}")
    all_imgs = []
    for root, _, files in os.walk(crop_base):
        for f in files:
            if f.endswith(".jpg"):
                all_imgs.append(os.path.join(root, f))

    total = len(all_imgs)
    if total == 0:
        print("No cropped images found to verify.")
        return

    chunk_size = (total + workers - 1) // workers
    chunks = [all_imgs[i : i + chunk_size] for i in range(0, total, chunk_size)]

    all_pure_black = 0
    all_scores = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify_chunk_worker, c) for c in chunks]
        for f in as_completed(futures):
            pb, sc = f.result()
            all_pure_black += pb
            all_scores.extend(sc)

    if not all_scores:
        print("WARNING: No images could be loaded for quality verification.")
        return

    scores_np = np.array(all_scores)
    very_blurry = int(np.sum(scores_np < 35))
    sharp_frames = int(np.sum(scores_np >= 50))
    mean_sharpness = np.mean(scores_np)
    median_sharpness = np.median(scores_np)

    print("==========================================")
    print("DATASET VERIFICATION RESULTS")
    print(f"Total frames verified: {total}")
    print(f"Pure black frames: {all_pure_black} ({all_pure_black / total * 100:.2f}%)")
    print(f"Mean Laplacian Sharpness: {mean_sharpness:.2f}")
    print(f"Median Laplacian Sharpness: {median_sharpness:.2f}")
    print(
        f"Sharp / Clear frames (Score >= 50): {sharp_frames} ({sharp_frames / total * 100:.2f}%)"
    )
    print(
        f"Noticeably blurry frames (Score < 35): {very_blurry} ({very_blurry / total * 100:.2f}%)"
    )
    print("==========================================\n")


# ==========================================
# Main Execution
# ==========================================


def main():
    args = parse_args()
    print("=" * 60)
    print("DMD Frame Extraction & Cropping Pipeline")
    print("=" * 60)
    print(f"DMD Video Directory:    {args.dmd_dir}")
    print(f"Raw Images Directory:   {args.out_raw_images}")
    print(f"Cropped Images Output:  {args.out_cropped}")
    print(f"Extraction FPS:         {args.sample_fps} fps")
    print(f"Crop Bounding Box:      {args.crop_box} (x, y, w, h)")
    print(f"Workers:                {args.workers}")
    print("=" * 60)

    if args.verify_only:
        verify_cropped_dataset(args.out_cropped, args.workers)
        return

    # 1. Extraction
    if not args.skip_extract:
        tasks = discover_rgb_face_videos(args.dmd_dir, args.out_raw_images)
        if not tasks:
            print(
                f"Warning: No rgb_face videos found in {args.dmd_dir}. Check directory path."
            )
        else:
            run_extraction(tasks, args.workers, args.sample_fps, args.jpeg_quality)

    # 2. Cropping
    if not args.skip_crop:
        crop_tasks, manifest_entries = collect_crop_tasks(
            args.out_raw_images, args.out_cropped
        )
        if not crop_tasks:
            print(
                f"Warning: No raw extracted images found in {args.out_raw_images} to crop."
            )
        else:
            run_cropping(
                crop_tasks,
                manifest_entries,
                args.out_cropped,
                args.workers,
                args.crop_box,
                args.jpeg_quality,
            )

    # 3. Verification
    if os.path.exists(args.out_cropped):
        verify_cropped_dataset(args.out_cropped, args.workers)

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
