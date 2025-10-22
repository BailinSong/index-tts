"""
测试padding和Conv1d输出长度
"""

import mlx.core as mx
import mlx.nn as nn

# 测试参数
batch = 1
seq_len = 30
in_channels = 512
out_channels = 1024
kernel_size = 5
dilation = 1

print(f"测试Conv1d输出长度")
print(f"输入: (batch={batch}, seq_len={seq_len}, channels={in_channels})")
print(f"卷积: kernel={kernel_size}, dilation={dilation}")

# 计算padding
effective_kernel_size = (kernel_size - 1) * dilation + 1
padding_total = effective_kernel_size - 1
padding_left = padding_total - padding_total // 2
padding_right = padding_total // 2

print(f"\nPadding计算:")
print(f"  effective_kernel_size = {effective_kernel_size}")
print(f"  padding_total = {padding_total}")
print(f"  padding_left = {padding_left}")
print(f"  padding_right = {padding_right}")

# 创建输入
x = mx.random.normal((batch, seq_len, in_channels))

# 手动padding
print(f"\n手动padding:")
print(f"  原始长度: {x.shape[1]}")

# 使用mlx_pad_reflect_1d的逻辑
if padding_left > 0:
    left_indices = mx.arange(1, padding_left + 1)
    left_pad = x[:, left_indices, :]
    x_padded = mx.concatenate([left_pad, x], axis=1)
else:
    x_padded = x

if padding_right > 0:
    current_len = x_padded.shape[1]
    original_end_idx = seq_len + (padding_left if padding_left > 0 else 0) - 1
    right_indices = mx.arange(original_end_idx - padding_right, original_end_idx)
    right_pad = x_padded[:, right_indices, :]
    x_padded = mx.concatenate([x_padded, right_pad], axis=1)

print(f"  Padding后长度: {x_padded.shape[1]}")
print(f"  预期: {seq_len + padding_left + padding_right}")

# 创建Conv1d (padding=0)
conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=0)

# Forward
output = conv(x_padded)
print(f"\nConv1d输出:")
print(f"  实际长度: {output.shape[1]}")
print(f"  预期长度: {seq_len}")  # 应该和输入一样

# 分析
if output.shape[1] == seq_len:
    print(f"✅ 长度正确！")
else:
    print(f"❌ 长度不对！实际{output.shape[1]} vs 预期{seq_len}")
    print(f"   差异: {output.shape[1] - seq_len}")

