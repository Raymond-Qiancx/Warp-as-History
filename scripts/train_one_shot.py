#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "/mnt/inspurfs/eb3d_t/share/hf_models/BestWishYsh/Helios-Mid"
DEFAULT_TARGET_ROOT = "/mnt/inspurfs/eb3d_t/wangyifan/sample/davis/DAVIS/JPEGImages/480p"
DEFAULT_TRAIN_CSV = "data/davis_camera_control_chunk1_single_car-roundabout_camctl23x_s16_t33_w34_prevframe_20260501.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the default one-shot Warp-as-History LoRA.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--transformer_path", default=None)
    parser.add_argument("--data_root", default="data/pi3x_first_frame_warp_480p")
    parser.add_argument("--target_root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=0)
    parser.add_argument("--preview_limit", type=int, default=0)
    parser.add_argument("--condition_fill", choices=["mean_first_frame", "black", "white", "first_frame", "none"], default="mean_first_frame")
    parser.add_argument("--visible_token_drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible_token_threshold", type=float, default=0.1)
    parser.add_argument("--direction_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse_probability", type=float, default=0.5)
    parser.add_argument(
        "--condition_variants",
        nargs="+",
        default=["warp", "tail_to_chunk", "first_tail_to_chunk"],
        choices=["warp", "tail_to_chunk", "first_tail_to_chunk"],
    )
    parser.add_argument("--previous_history_probability", type=float, default=0.5)
    parser.add_argument("--previous_history_sizes", type=int, nargs=3, default=[6, 2, 1])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend_script", default=None)
    parser.add_argument("--keep_backend_config", action="store_true")
    return parser.parse_args()


def resolve_backend_script(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    candidates.extend(
        [
            Path.cwd() / "scripts" / "my" / "train_davis_reverse_history_lora_exact.py",
            Path("/mnt/hwfile/wangyifan/code/Helios/scripts/my/train_davis_reverse_history_lora_exact.py"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find the compatible Helios training backend. Pass --backend_script.")


def build_backend_command(args: argparse.Namespace, backend_script: Path) -> list[str]:
    transformer_path = args.transformer_path or args.model_path
    command = [
        sys.executable,
        str(backend_script),
        "--base_model_path",
        str(args.model_path),
        "--transformer_path",
        str(transformer_path),
        "--data_root",
        str(args.data_root),
        "--gt_root",
        str(args.target_root),
        "--prompt_csv",
        str(args.train_csv),
        "--output_dir",
        str(args.output_dir),
        "--limit",
        str(int(args.sample_count)),
        "--max_steps",
        str(int(args.max_steps)),
        "--lr",
        str(float(args.learning_rate)),
        "--lr_schedule",
        "constant",
        "--lr_schedule_final_ratio",
        "1.0",
        "--optimizer",
        "adamw",
        "--adamw_weight_decay",
        "0.01",
        "--warmup_steps",
        "20",
        "--max_grad_norm",
        "1.0",
        "--seed",
        str(int(args.seed)),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        "33",
        "--num_latent_frames_per_chunk",
        "9",
        "--history_sizes",
        "1",
        "1",
        "9",
        "--history_temporal_layout",
        "long_mid_short",
        "--pyramid_num_inference_steps_list",
        "20",
        "20",
        "20",
        "--flow_matching_stage_sampling",
        "fixed",
        "--flow_matching_stage_id",
        "0",
        "--flow_matching_train_exact_timestep_sampling",
        "training_density",
        "--guidance_scale",
        "5.0",
        "--cfg_zero_star",
        "auto",
        "--zero_steps",
        "1",
        "--warp_name",
        "warp_first_to_all_mesh_noedge_topfarfill.mp4",
        "--mask_name",
        "visibility_mask_mesh_noedge_topfarfill.mp4",
        "--history_packing",
        "nonreverse_rgbvae",
        "--reverse_history_rope_remap",
        "last_n_same_order",
        "--reverse_history_rope_remap_count",
        "9",
        "--reverse_history_rope_remap_delta",
        "0",
        "--warp_invisible_fill",
        str(args.condition_fill),
        "--warp_condition_mode",
        "history",
        "--history_prefix_mode",
        "local",
        "--train_prev_gt_history_prob",
        str(float(args.previous_history_probability)),
        "--train_prev_gt_history_sizes",
        *(str(int(x)) for x in args.previous_history_sizes),
        "--lora_rank",
        str(int(args.rank)),
        "--lora_alpha",
        str(int(args.alpha)),
        "--lora_dropout",
        str(float(args.dropout)),
        "--lora_target_modules",
        "attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0",
        "--lora_adapter_name",
        "warp_as_history",
        "--save_every",
        str(int(args.save_every)),
        "--log_every",
        str(int(args.log_every)),
        "--preview_every",
        str(int(args.preview_every)),
        "--preview_limit",
        str(int(args.preview_limit)),
        "--prompt_cache_dir",
        "data/prompt_cache/helios_mid_512",
        "--lazy_items",
        "--offload_items_to_cpu",
        "--item_cache_size",
        "0",
        "--shuffle",
        "--no_initial_preview",
    ]
    command.append("--history_visible_token_drop" if args.visible_token_drop else "--no-history_visible_token_drop")
    command.extend(["--history_visible_token_threshold", str(float(args.visible_token_threshold))])
    if args.direction_augmentation:
        command.extend(
            [
                "--temporal_bidirectional_aug",
                "--temporal_bidirectional_reverse_prob",
                str(float(args.reverse_probability)),
                "--temporal_bidirectional_warp_variants",
                *(str(item) for item in args.condition_variants),
            ]
        )
    command.append("--tensorboard" if args.tensorboard else "--no-tensorboard")
    command.append("--gradient_checkpointing" if args.gradient_checkpointing else "--no-gradient_checkpointing")
    if args.overwrite:
        command.append("--overwrite")
    return command


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_clean_config(args: argparse.Namespace, output_dir: Path, backend_script: Path) -> None:
    config = {
        "model": {
            "base": str(args.model_path),
            "transformer": str(args.transformer_path or args.model_path),
        },
        "data": {
            "root": str(args.data_root),
            "target_root": str(args.target_root),
            "train_csv": str(args.train_csv),
            "sample_count": int(args.sample_count),
        },
        "training": {
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "height": int(args.height),
            "width": int(args.width),
            "learning_rate": float(args.learning_rate),
            "optimizer": "adamw",
            "warmup_steps": 20,
            "max_grad_norm": 1.0,
            "stage": "stage0",
            "history_sizes": [1, 1, 9],
            "previous_history_probability": float(args.previous_history_probability),
            "previous_history_sizes": [int(x) for x in args.previous_history_sizes],
            "condition_fill": str(args.condition_fill),
            "visible_token_drop": bool(args.visible_token_drop),
            "visible_token_threshold": float(args.visible_token_threshold),
            "direction_augmentation": bool(args.direction_augmentation),
            "reverse_probability": float(args.reverse_probability),
            "condition_variants": [str(x) for x in args.condition_variants],
        },
        "lora": {
            "rank": int(args.rank),
            "alpha": int(args.alpha),
            "dropout": float(args.dropout),
            "target_modules": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"],
        },
        "runtime": {
            "backend_script": str(backend_script),
        },
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_clean_loss(output_dir: Path) -> None:
    source = output_dir / "train_loss.json"
    if not source.is_file():
        return
    records = json.loads(source.read_text(encoding="utf-8"))
    clean_records = []
    for record in records:
        clean_records.append(
            {
                "step": int(record["step"]),
                "sample": str(record["seq"]),
                "loss": float(record["loss"]),
                "learning_rate": float(record["lr"]),
                "gradient_norm": None if record.get("grad_norm") is None else float(record["grad_norm"]),
                "used_previous_history": bool(record.get("used_prev_gt_history", False)),
                "selected_previous_chunks": [int(x) for x in record.get("selected_prev_gt_chunk_starts", [])],
            }
        )
    (output_dir / "loss.json").write_text(json.dumps(clean_records, indent=2) + "\n", encoding="utf-8")


def normalize_outputs(args: argparse.Namespace, backend_script: Path) -> None:
    output_dir = Path(args.output_dir)
    write_clean_config(args, output_dir, backend_script)
    write_clean_loss(output_dir)

    for checkpoint in sorted(output_dir.glob("visible_lora_state_step*.pt")):
        step_name = checkpoint.name.replace("visible_lora_state", "lora_state", 1)
        copy_if_present(checkpoint, output_dir / step_name)
        checkpoint.unlink()
    copy_if_present(output_dir / "visible_lora_state.pt", output_dir / "lora_state.pt")
    if (output_dir / "visible_lora_state.pt").is_file():
        (output_dir / "visible_lora_state.pt").unlink()

    if not args.keep_backend_config:
        for legacy_name in ("train_config.json", "train_loss.json"):
            legacy_path = output_dir / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()


def main() -> None:
    args = parse_args()
    backend_script = resolve_backend_script(args.backend_script)
    command = build_backend_command(args, backend_script)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "train_one_shot_start", "output_dir": str(output_dir)}), flush=True)
    subprocess.run(command, check=True)
    normalize_outputs(args, backend_script)
    print(json.dumps({"event": "train_one_shot_done", "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
