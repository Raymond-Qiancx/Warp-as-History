# Warp-as-History

## Environment Setup

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/yyfz/Warp-as-History.git
cd Warp-as-History
```

If you have already cloned the repository without submodules, run:

```bash
git submodule update --init --recursive
```

The Pi3 dependency is tracked as a Git submodule under `third_party/Pi3`.

## Pi3X Camera Warp

The main pipeline can either consume a pre-rendered `warp_video` or render one
from `camera_poses` with Pi3X:

```python
from warp_as_history import WarpAsHistoryPipeline

pipe = WarpAsHistoryPipeline.from_pretrained(...).to("cuda")
video = pipe(
    prompt="a car driving through a roundabout",
    image=first_frame,
    camera_poses=camera_poses,
    pi3x_ckpt="/path/to/Pi3X/model.safetensors",
    lora_path="/path/to/visible_lora_state_step1000.pt",
)
```

If `pi3x_ckpt` is omitted, the renderer uses `PI3X_CKPT` when set, otherwise
it falls back to `Pi3X.from_pretrained("yyfz233/Pi3X")`. The default Pi3 repo is
`third_party/Pi3`.

## One-Shot Training

The default one-shot LoRA recipe is exposed as a single release entry point:

```bash
python scripts/train_one_shot.py \
  --output_dir runs/car_roundabout_one_shot \
  --data_root data/pi3x_first_frame_warp_480p \
  --target_root /path/to/DAVIS/JPEGImages/480p \
  --train_csv data/davis_camera_control_chunk1_single_car-roundabout_camctl23x_s16_t33_w34_prevframe_20260501.csv
```

The script writes `lora_state*.pt`, `loss.json`, and `training_config.json`.
