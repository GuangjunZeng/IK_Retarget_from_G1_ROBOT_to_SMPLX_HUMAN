#!/usr/bin/env python3
"""Quick validator for data structure of outputs from 29dof_npz_to_complete_npz.py."""

import argparse
import pathlib
import sys
from typing import Dict, Iterable, Tuple

import numpy as np

# Example usage:
# python scripts/verify_npz_robot.py  --input /home/retarget/workbench/server3_data/locomotion/human/ik_based/npz/000001.npz  --reference /home/retarget/workbench/server3_data/locomotion/reference/000001.npz   



def load_npz(path: pathlib.Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def format_shape(array: np.ndarray) -> str:
    return "x".join(str(dim) for dim in array.shape) if array.shape else "scalar"


def summarize_array(name: str, array: np.ndarray) -> str:
    shape = format_shape(array)
    dtype = array.dtype
    summary = f"{name}: shape={shape}, dtype={dtype}"
    if array.ndim == 1 and array.size <= 10:
        summary += f", values={array.tolist()}"
    if array.ndim == 0:
        summary += f", value={array.item()}"
    return summary


def validate_structure(data: Dict[str, np.ndarray]) -> Iterable[str]:
    # Validate only keys that actually exist in the NPZ
    frames = None

    if "pose_body" in data:
        pose_body = np.asarray(data["pose_body"]) 
        if pose_body.ndim != 2:
            yield "pose_body must be 2-D (frames x parameters)"
        else:
            frames = pose_body.shape[0]
            if pose_body.shape[1] % 3 != 0:
                yield "pose_body second dimension must be divisible by 3 (axis-angle triplets)"

    if "root_orient" in data and frames is not None:
        root_orient = np.asarray(data["root_orient"]) 
        if root_orient.shape != (frames, 3):
            yield "root_orient must have shape (num_frames, 3)"

    if "trans" in data and frames is not None:
        trans = np.asarray(data["trans"]) 
        if trans.shape != (frames, 3):
            yield "trans must have shape (num_frames, 3)"

    if "betas" in data:
        betas = np.asarray(data["betas"]) 
        if betas.ndim not in (1, 2):
            yield "betas should be 1-D or 2-D"
        if betas.ndim == 2 and betas.shape[0] != 1:
            yield "betas 2-D array expected to have shape (1, num_betas)"

    if "gender" in data:
        gender = data["gender"] 
        if np.asarray(gender).size != 1:
            yield "gender should be a scalar or single string"

    if "mocap_frame_rate" in data:
        frame_rate = np.asarray(data["mocap_frame_rate"]) 
        if frame_rate.size != 1:
            yield "mocap_frame_rate should be a scalar"

#对比每一个
def compare_structures(source: Dict[str, np.ndarray], reference: Dict[str, np.ndarray]) -> Iterable[str]:
    if set(source.keys()) != set(reference.keys()):
        missing = set(reference.keys()) - set(source.keys())
        extra = set(source.keys()) - set(reference.keys())
        if missing:
            yield f"Compared NPZ is missing keys: {sorted(missing)}"
        if extra:
            yield f"Compared NPZ has extra keys: {sorted(extra)}"

    for key in set(source.keys()) & set(reference.keys()):
        src_shape = np.asarray(source[key]).shape
        ref_shape = np.asarray(reference[key]).shape
        if src_shape != ref_shape:
            yield f"Shape mismatch for key '{key}': {src_shape} vs {ref_shape}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the structure of an SMPL-X NPZ file produced from robot data."
    )
    parser.add_argument("--input", required=True, help="Path to NPZ file to validate")
    parser.add_argument(
        "--reference",
        help="Optional reference NPZ to compare keys and shapes (e.g. a known-good file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = pathlib.Path(args.input).expanduser()
    data = load_npz(input_path)

    print(f"Inspecting: {input_path}")
    print("Keys:", sorted(list(data.keys())))
    for key in sorted(data.keys()):
        print("  " + summarize_array(key, np.asarray(data[key])))

    errors = list(validate_structure(data))
    if errors:
        print("\n❌ Structural issues detected:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✅ Structure matches expected SMPLX-style layout.")

    if args.reference:
        reference_path = pathlib.Path(args.reference).expanduser()
        ref_data = load_npz(reference_path)
        print(f"\nComparing against reference: {reference_path}")
        compare_errors = list(compare_structures(data, ref_data))
        if compare_errors:
            print("❌ Differences found:")
            for err in compare_errors:
                print(f"  - {err}")
            errors.extend(compare_errors)
        else:
            print("✅ Keys and shapes align with the reference file.")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()


