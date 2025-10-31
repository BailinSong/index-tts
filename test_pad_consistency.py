import numpy as np
import torch

from indextts.s2mel.modules.mlx_wavenet_model import pad1d_mlx
import mlx.core as mx


def run_case(T: int, C: int, left: int, right: int, mode: str):
    B = 2
    x_np = np.random.randn(B, T, C).astype(np.float32)

    # Torch reference on (B,C,T)
    x_torch = torch.from_numpy(x_np.transpose(0, 2, 1))  # (B,C,T)
    if mode == 'reflect':
        length = x_torch.shape[-1]
        max_pad = max(left, right)
        extra = 0
        if length <= max_pad:
            extra = max_pad - length + 1
            if extra > 0:
                x_torch = torch.nn.functional.pad(x_torch, (0, extra), mode='constant', value=0.0)
        x_pad_t = torch.nn.functional.pad(x_torch, (left, right), mode='reflect')
        if extra > 0:
            x_pad_t = x_pad_t[..., : x_pad_t.shape[-1] - extra]
    elif mode == 'zero':
        x_pad_t = torch.nn.functional.pad(x_torch, (left, right), mode='constant', value=0.0)
    else:
        x_pad_t = torch.nn.functional.pad(x_torch, (left, right), mode='constant', value=0.0)
    x_pad_ref = x_pad_t.numpy().transpose(0, 2, 1)  # (B,T,C)

    # MLX under test on (B,T,C)
    x_mx = mx.array(x_np)
    x_pad_mx = pad1d_mlx(x_mx, (left, right), mode=mode)
    x_pad_mx_np = np.array(x_pad_mx)

    same_shape = x_pad_mx_np.shape == x_pad_ref.shape
    max_diff = np.max(np.abs(x_pad_mx_np - x_pad_ref)) if same_shape else float('inf')
    mean_diff = float(np.mean(np.abs(x_pad_mx_np - x_pad_ref))) if same_shape else float('inf')

    print(f"Case T={T},C={C},L={left},R={right},mode={mode} -> shape_ref={x_pad_ref.shape}, shape_mlx={x_pad_mx_np.shape}")
    if not same_shape:
        print("  ❌ shape mismatch")
    else:
        print(f"  max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}")
    return same_shape, max_diff, mean_diff


def main():
    np.random.seed(0)
    cases = [
        # typical
        (415, 512, 2, 2, 'reflect'),
        (415, 512, 2, 2, 'zero'),
        # edge: small T with large pad (reflect pre-extend path)
        (3, 4, 5, 1, 'reflect'),
        (2, 8, 3, 3, 'reflect'),
        # asymmetric
        (17, 16, 3, 1, 'reflect'),
        (17, 16, 3, 1, 'zero'),
        # larger
        (128, 64, 7, 5, 'reflect'),
    ]

    all_ok = True
    for T, C, L, R, mode in cases:
        same, maxd, _ = run_case(T, C, L, R, mode)
        if (not same) or (maxd > 1e-6):
            all_ok = False

    print("\nSummary: ")
    print("  ✅ All cases matched" if all_ok else "  ❌ Some cases mismatched")


if __name__ == "__main__":
    main()


