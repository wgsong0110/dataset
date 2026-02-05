#!/usr/bin/env python3
"""
COLMAP sparse reconstruction → NeRF Blender format converter.

Reads COLMAP binary files (cameras.bin, images.bin) and produces
transforms_train.json / transforms_test.json / transforms_val.json
with images symlinked into train/ test/ val/ subdirectories.

Usage:
    python colmap2blender.py <input_dir> <output_dir> [--test_every 8]

    input_dir  : directory containing images/ and sparse/0/
    output_dir : where to write the blender-format dataset
"""

import argparse
import json
import math
import os
import shutil
import struct
import sys

import numpy as np


# ─── COLMAP binary readers ───────────────────────────────────────

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),           # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),   # fx, fy, cx, cy, k1, k2, k3, k4
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))   # tx, ty, tz
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            # Skip 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # Each point2D: x(double), y(double), point3D_id(int64) = 24 bytes
            f.read(24 * num_points2D)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


# ─── Math helpers ────────────────────────────────────────────────

def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ])
    return R


def get_c2w(qvec, tvec):
    """
    COLMAP stores world-to-camera (R, t).
    Returns camera-to-world 4x4 matrix in OpenGL/Blender convention.
    """
    R = qvec2rotmat(qvec)
    t = np.array(tvec).reshape(3, 1)

    # world-to-camera
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t

    # camera-to-world
    c2w = np.linalg.inv(w2c)

    # COLMAP: x-right, y-down, z-forward
    # OpenGL/Blender: x-right, y-up, z-backward
    # Flip y and z axes
    c2w[:3, 1:3] *= -1

    return c2w


def focal_to_fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))


def get_focal(camera):
    """Extract fx, fy from COLMAP camera params."""
    model = camera["model"]
    params = camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
        fx = fy = params[0]
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy = params[0], params[1]
    elif model == "FOV":
        fx = fy = params[0]
    else:
        raise ValueError(f"Unknown camera model: {model}")
    return fx, fy


# ─── Main conversion ────────────────────────────────────────────

def convert(input_dir, output_dir, test_every=8):
    sparse_dir = os.path.join(input_dir, "sparse", "0")
    images_dir = os.path.join(input_dir, "images")

    if not os.path.isdir(sparse_dir):
        print(f"ERROR: {sparse_dir} not found")
        return False
    if not os.path.isdir(images_dir):
        print(f"ERROR: {images_dir} not found")
        return False

    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    print(f"  cameras: {len(cameras)}, images: {len(images)}")

    # Sort images by name for deterministic split
    sorted_imgs = sorted(images.values(), key=lambda x: x["name"])

    # Get camera FOV (use first camera)
    first_cam = cameras[sorted_imgs[0]["camera_id"]]
    fx, fy = get_focal(first_cam)
    w, h = first_cam["width"], first_cam["height"]
    camera_angle_x = focal_to_fov(fx, w)
    camera_angle_y = focal_to_fov(fy, h)

    # Build frames
    all_frames = []
    for img in sorted_imgs:
        cam = cameras[img["camera_id"]]
        c2w = get_c2w(img["qvec"], img["tvec"])
        frame = {
            "file_path": img["name"],
            "transform_matrix": c2w.tolist(),
        }
        # Per-frame intrinsics if multi-camera
        if len(cameras) > 1:
            _fx, _fy = get_focal(cam)
            _w, _h = cam["width"], cam["height"]
            frame["camera_angle_x"] = focal_to_fov(_fx, _w)
            frame["camera_angle_y"] = focal_to_fov(_fy, _h)
            frame["w"] = _w
            frame["h"] = _h
        all_frames.append(frame)

    # Split: every test_every-th image → test, next → val, rest → train
    train_frames, test_frames, val_frames = [], [], []
    for i, frame in enumerate(all_frames):
        if i % test_every == 0:
            test_frames.append(frame)
        elif i % test_every == 1:
            val_frames.append(frame)
        else:
            train_frames.append(frame)

    # If val is empty (very few images), use test as val too
    if not val_frames:
        val_frames = test_frames

    # Create output directories
    for split in ("train", "test", "val"):
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Copy images and update file_path
    def process_split(frames, split):
        for frame in frames:
            src = os.path.join(images_dir, frame["file_path"])
            # Use original filename
            fname = os.path.basename(frame["file_path"])
            dst = os.path.join(output_dir, split, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                # Symlink to save space
                os.symlink(os.path.abspath(src), dst)
            frame["file_path"] = f"./{split}/{fname}"

    process_split(train_frames, "train")
    process_split(test_frames, "test")
    process_split(val_frames, "val")

    # Write transforms JSONs
    base = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "w": int(w),
        "h": int(h),
    }

    for split, frames in [("train", train_frames), ("test", test_frames), ("val", val_frames)]:
        out = {**base, "frames": frames}
        path = os.path.join(output_dir, f"transforms_{split}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"  train: {len(train_frames)}, test: {len(test_frames)}, val: {len(val_frames)}")
    print(f"  → {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="COLMAP → Blender format converter")
    parser.add_argument("input_dir", help="COLMAP dataset dir (contains images/ and sparse/0/)")
    parser.add_argument("output_dir", help="Output directory for blender format")
    parser.add_argument("--test_every", type=int, default=8, help="Every N-th image for test split")
    args = parser.parse_args()

    print(f"Converting: {args.input_dir}")
    ok = convert(args.input_dir, args.output_dir, args.test_every)
    if ok:
        print("Done!")
    else:
        print("Failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
