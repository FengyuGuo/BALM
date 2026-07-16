#!/usr/bin/env python3
"""Align a binary PCD's ground plane with +Z without altering non-XYZ fields."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def read_binary_pcd(path: Path) -> tuple[bytes, np.ndarray, bytes]:
    """Read a binary PCD as raw structured records, preserving every field."""
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no DATA declaration")
            header_lines.append(line)
            if line.strip().lower().startswith(b"data"):
                if line.strip().lower() != b"data binary":
                    raise ValueError("Only uncompressed 'DATA binary' PCD files are supported")
                break
        payload = file.read()

    header = b"".join(header_lines)
    values = {
        line.split(maxsplit=1)[0].upper(): line.split(maxsplit=1)[1].split()
        for line in (raw.decode("ascii").strip() for raw in header_lines)
        if line and not line.startswith("#") and len(line.split(maxsplit=1)) == 2
    }
    fields = values["FIELDS"]
    sizes = [int(item) for item in values["SIZE"]]
    types = values["TYPE"]
    counts = [int(item) for item in values.get("COUNT", ["1"] * len(fields))]
    points_count = int(values["POINTS"][0])

    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("The PCD must contain x, y, and z fields")

    type_codes = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    }
    try:
        dtype_fields = []
        for field, size, field_type, count in zip(fields, sizes, types, counts):
            item = (field, type_codes[(field_type, size)])
            dtype_fields.append(item if count == 1 else (*item, (count,)))
        dtype = np.dtype(dtype_fields)
    except KeyError as error:
        raise ValueError(f"Unsupported PCD field type/size: {error.args[0]}") from error

    expected_size = points_count * dtype.itemsize
    if len(payload) < expected_size:
        raise ValueError(f"Expected at least {expected_size} data bytes, got {len(payload)}")

    # Some PCD exporters append opaque bytes after the declared point records.
    # Open3D ignores them; retain them verbatim instead of rejecting the file.
    records = np.frombuffer(payload[:expected_size], dtype=dtype).copy()
    return header, records, payload[expected_size:]


def write_binary_pcd(path: Path, header: bytes, records: np.ndarray, trailing: bytes) -> None:
    """Write the original header and raw records without serializing attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header)
        file.write(records.tobytes())
        file.write(trailing)


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation that maps the unit vector source to target."""
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)

    if sine < 1e-12:
        # Vectors are already aligned, or point in exactly opposite directions.
        if cosine > 0:
            return np.eye(3)
        # Rotate 180 degrees about any axis orthogonal to source.
        axis = np.zeros(3)
        axis[np.argmin(np.abs(source))] = 1.0
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    skew = np.array(
        [[0.0, -cross[2], cross[1]],
         [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the dominant plane, rotate its normal onto +Z, and save the complete cloud."
    )
    parser.add_argument("input_pcd", type=Path, help="Input PCD file")
    parser.add_argument("output_pcd", type=Path, help="Aligned output PCD file")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.05,
        help="Maximum point-to-plane distance in metres (default: 0.05)",
    )
    parser.add_argument("--ransac-n", type=int, default=3, help="Points per RANSAC hypothesis")
    parser.add_argument("--iterations", type=int, default=1000, help="RANSAC iterations")
    args = parser.parse_args()

    header, records, trailing = read_binary_pcd(args.input_pcd)
    if len(records) == 0:
        raise ValueError(f"Could not read points from {args.input_pcd}")

    # Use Open3D only to find ground inliers.  The raw PCD records are used for
    # output, so intensity and every other non-coordinate byte stay untouched.
    positions = np.column_stack((records["x"], records["y"], records["z"])).astype(float)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(positions))

    # RANSAC returns the dominant plane ax + by + cz + d = 0 and its inliers.
    plane, inliers = cloud.segment_plane(
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.iterations,
    )
    plane = np.asarray(plane)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)

    # Choose the positive normal so the transformed plane faces +Z.
    if normal[2] < 0:
        normal = -normal
    rotation = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0]))

    # Change only x/y/z. Every other structured field, including intensity,
    # remains byte-for-byte identical to its input value.
    aligned_plane_offset = plane[3] if plane[2] >= 0 else -plane[3]
    aligned_positions = positions @ rotation.T
    aligned_positions[:, 2] += aligned_plane_offset
    records["x"] = aligned_positions[:, 0]
    records["y"] = aligned_positions[:, 1]
    records["z"] = aligned_positions[:, 2]

    write_binary_pcd(args.output_pcd, header, records, trailing)

    print(f"Plane: {plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0")
    print(f"Ground inliers: {len(inliers)} / {len(records)}")
    print(f"Saved aligned cloud: {args.output_pcd}")


if __name__ == "__main__":
    main()
