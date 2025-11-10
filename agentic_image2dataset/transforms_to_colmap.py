"""
Converter from NeRF-style transforms.json to COLMAP format for gsplat compatibility.

This module converts the transforms.json format (used by Stable Virtual Camera)
to COLMAP sparse reconstruction format that gsplat expects.
"""

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np


def transforms_json_to_colmap(
    transforms_json_path: Path,
    images_dir: Path,
    output_colmap_dir: Path,
    image_prefix: str = "view_",
) -> Dict[str, any]:
    """
    Convert NeRF-style transforms.json to COLMAP format.

    Args:
        transforms_json_path: Path to transforms.json file
        images_dir: Directory containing the images
        output_colmap_dir: Directory to save COLMAP files
        image_prefix: Prefix for image filenames (default: "view_")

    Returns:
        Dictionary with success status and output directory
    """
    output_colmap_dir = Path(output_colmap_dir)
    output_colmap_dir.mkdir(parents=True, exist_ok=True)

    # Load transforms.json
    with open(transforms_json_path, "r") as f:
        transforms_data = json.load(f)

    frames = transforms_data["frames"]
    num_images = len(frames)

    # Create COLMAP cameras.txt
    cameras_file = output_colmap_dir / "cameras.txt"
    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        # Use the first frame to determine camera parameters
        # All frames should have the same intrinsics in this case
        first_frame = frames[0]
        w = int(first_frame["w"])
        h = int(first_frame["h"])
        fx = float(first_frame["fl_x"])
        fy = float(first_frame["fl_y"])
        cx = float(first_frame["cx"])
        cy = float(first_frame["cy"])

        # COLMAP PINHOLE format: CAMERA_ID, PINHOLE, WIDTH, HEIGHT, FX, FY, CX, CY
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # Create COLMAP images.txt
    images_file = output_colmap_dir / "images.txt"
    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}\n")

        for i, frame in enumerate(frames):
            # Get transform matrix (camera-to-world)
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            # Convert to world-to-camera
            w2c = np.linalg.inv(c2w)

            # Extract rotation and translation
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            # Convert rotation matrix to quaternion
            # COLMAP uses quaternion format: w, x, y, z
            trace = np.trace(R)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s=4*qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s=4*qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s=4*qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s=4*qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

            # Get image filename
            file_path = frame.get("file_path", f"{image_prefix}{i:04d}.png")
            if file_path.startswith("./"):
                file_path = file_path[2:]
            image_name = os.path.basename(file_path)

            # Write image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            image_id = i + 1
            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {image_name}\n"
            )
            # Write empty points line (no 3D points from transforms.json alone)
            f.write("\n")

    # Create empty points3D.txt (no 3D points available from transforms.json)
    points3d_file = output_colmap_dir / "points3D.txt"
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        f.write("# Number of points: 0\n")

    return {
        "success": True,
        "colmap_dir": output_colmap_dir,
        "num_images": num_images,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert transforms.json to COLMAP format"
    )
    parser.add_argument("transforms_json", type=str, help="Path to transforms.json")
    parser.add_argument("images_dir", type=str, help="Directory containing images")
    parser.add_argument("output_dir", type=str, help="Output COLMAP directory")
    args = parser.parse_args()

    result = transforms_json_to_colmap(
        Path(args.transforms_json),
        Path(args.images_dir),
        Path(args.output_dir),
    )
    print(f"Conversion successful: {result['num_images']} images")
    print(f"COLMAP files saved to: {result['colmap_dir']}")
