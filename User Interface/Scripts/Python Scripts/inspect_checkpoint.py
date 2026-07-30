"""
One-off diagnostic: prints what's actually inside agent0.pt so we can
confirm the real network architecture and pick the right key to load.
Run this once, paste the output back.
"""
import torch

for name in ("agent0.pt", "agent1.pt"):
    print(f"\n=== {name} ===")
    ckpt = torch.load(name, map_location="cpu")
    print("top-level keys:", list(ckpt.keys()))

    for meta_key in ("schema", "obs_dim", "n_actions", "hidden", "actions",
                      "channels", "globals", "policy", "action_safety"):
        if meta_key in ckpt:
            print(f"  {meta_key}: {ckpt[meta_key]}")

    if "net" in ckpt:
        print("  net sub-keys:")
        for k, v in ckpt["net"].items():
            shape = tuple(v.shape) if hasattr(v, "shape") else v
            print(f"    {k}: {shape}")
