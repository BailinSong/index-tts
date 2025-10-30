"""
MLX Implementation of WaveNet for DiT Final Layer
使用经过测试验证的SConv1d实现
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import typing as tp


def init_linear_pytorch_compatible(linear_layer: nn.Linear, std: float = 0.01):
    """
    Initialize MLX Linear layer to match PyTorch initialization.
    
    Args:
        linear_layer: MLX Linear layer
        std: Standard deviation for normal initialization (default: 0.01)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Linear layers
    linear_layer.weight = mx.random.normal(linear_layer.weight.shape) * std
    # MLX Linear layers don't have bias attribute, they have bias parameter
    if hasattr(linear_layer, 'bias') and linear_layer.bias is not None:
        linear_layer.bias = mx.zeros_like(linear_layer.bias)


def init_conv_pytorch_compatible(conv_layer: nn.Conv1d, std: float = 0.01):
    """
    Initialize MLX Conv1d layer to match PyTorch initialization.
    
    Args:
        conv_layer: MLX Conv1d layer
        std: Standard deviation for normal initialization (default: 0.01)
    """
    # PyTorch uses normal_(mean=0.0, std=std) for Conv1d layers
    conv_layer.weight = mx.random.normal(conv_layer.weight.shape) * std
    # MLX Conv1d layers don't have bias attribute, they have bias parameter
    if hasattr(conv_layer, 'bias') and conv_layer.bias is not None:
        conv_layer.bias = mx.zeros_like(conv_layer.bias)


def get_extra_padding_for_conv1d_mlx(x: mx.array, kernel_size: int, stride: int,
                                      padding_total: int = 0) -> int:
    """MLX版本的get_extra_padding_for_conv1d，完全复刻PyTorch版本"""
    length = x.shape[1]  # MLX格式: (batch, seq_len, channels)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)

def pad1d_mlx(x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """严格复刻 PyTorch F.pad 1D 行为（以时间维为第二维），纯 MLX/NumPy 实现。
    支持 mode='reflect' 与 'zero'/其他常量模式；在 reflect 模式下复刻 Meta 的 pad1d：
    当 T <= max(left,right) 时先在右侧补零到允许反射的最小长度，再反射，最后裁剪回目标长度 L+T+R。
    """
    T = int(x.shape[1])  # (B, T, C)
    left, right = int(paddings[0]), int(paddings[1])
    assert left >= 0 and right >= 0, (left, right)

    if mode != 'reflect':
        # 常量填充（含 zero）：直接使用 mx.pad
        if left == 0 and right == 0:
            return x
        return mx.pad(x, ((0, 0), (left, right), (0, 0)), mode='constant', constant_values=float(value))

    # reflect 模式
    max_pad = left if left > right else right
    extra_pad = 0
    x_ext = x
    if T <= max_pad:
        # 右侧先补零，匹配 torch pad1d 的前置修复以允许反射
        extra_pad = max_pad - T + 1
        if extra_pad > 0:
            x_ext = mx.pad(x_ext, ((0, 0), (0, extra_pad), (0, 0)), mode='constant', constant_values=0.0)
    Tp = int(x_ext.shape[1])  # 反射实际长度

    # 目标长度（去掉 extra_pad 后）
    target_len = left + T + right
    total_len = left + Tp + right

    # 生成反射索引（不使用 Torch）
    # 反射公式：周期 P = 2*Tp-2；i' = abs(i) % P；若 i' >= Tp: i'' = 2*Tp-2 - i'，否则 i''=i'
    import numpy as np
    P = 2 * Tp - 2
    # 构造基础位置：[-left, ..., -1, 0, ..., Tp-1, Tp, ..., Tp+right-1]
    base = np.arange(total_len, dtype=np.int64) - left
    if P > 0:
        imod = np.abs(base) % P
        idx_np = np.where(imod >= Tp, (2 * Tp - 2) - imod, imod).astype(np.int64)
    else:
        # Tp==1 的退化情形（理论上 reflect 不合法，保持索引为 0）
        idx_np = np.zeros_like(base, dtype=np.int64)

    idx_mx = mx.array(idx_np.astype(np.int32))
    x_padded_full = mx.take(x_ext, idx_mx, axis=1)

    # 裁剪掉因 extra_pad 引入的冗余，使最终时间维为 L+T+R
    if x_padded_full.shape[1] != target_len:
        x_padded_full = x_padded_full[:, :target_len, :]

    # 详细日志：定位 419 来源
    try:
        from indextts.utils.cfm_debugger import log_cfm_stage
        log_cfm_stage(
            "wavenet_l0_sconv_pad_internal",
            mlx_data={
                'left': int(left),
                'right': int(right),
                'extra_pad': int(extra_pad),
                'T_in': int(T),
                'Tp_after_extra': int(Tp),
                'total_len_index': int(total_len),
                'target_len_crop': int(target_len),
                'len_before_crop': int(mx.take(x_ext, idx_mx, axis=1).shape[1]),
                'len_after_crop': int(x_padded_full.shape[1])
            },
            layer=0,
        )
    except Exception:
        pass

    return x_padded_full

def mlx_pad_reflect_1d(x, padding_left, padding_right):
    """
    MLX实现的reflect padding for 1D，完全复刻PyTorch F.pad的行为
    x: (batch, seq_len, channels)
    """
    return pad1d_mlx(x, (padding_left, padding_right), mode='reflect')


class MLXNormConv1d(nn.Module):
    """MLX版本的NormConv1d，完全复刻PyTorch版本"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                             dilation=dilation, groups=groups, bias=bias)
        init_conv_pytorch_compatible(self.conv)

    def __call__(self, x):
        return self.conv(x)

class MLXSConv1d(nn.Module):
    """MLX版本的SConv1d，完全复刻PyTorch版本 - 经过测试验证的实现"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect', **kwargs):
        super().__init__()
        self.kernel_size_init = kernel_size  # Store original kernel_size
        self.stride_init = stride
        self.dilation_init = dilation
        self.causal = causal
        self.pad_mode = pad_mode
        # For debugging which WaveNet layer this conv belongs to
        self.debug_layer_index = -1
        
        # MLX NormConv1d equivalent
        self.conv = MLXNormConv1d(in_channels, out_channels, kernel_size, stride,
                                  dilation=dilation, groups=groups, bias=bias, causal=causal,
                                  norm=norm, norm_kwargs=norm_kwargs)

    def get_extra_padding_for_conv1d(self, x: mx.array, kernel_size: int, stride: int,
                                     padding_total: int = 0) -> int:
        """MLX版本的get_extra_padding_for_conv1d"""
        return get_extra_padding_for_conv1d_mlx(x, kernel_size, stride, padding_total)

    def mlx_pad1d(self, x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
        """MLX版本的pad1d，模拟PyTorch F.pad的行为"""
        return pad1d_mlx(x, paddings, mode, value)

    def __call__(self, x):
        # MLX 格式: (batch, seq_len, channels)
        batch, seq_len, channels = x.shape
    
        # 使用构造时记录的原始配置，严格对齐 PyTorch 模型定义
        kernel_size_raw = int(self.kernel_size_init)
        try:
            # 若权重维度与初始化不一致，取较小者以对齐 Torch 行为
            kw = int(self.conv.conv.weight.shape[1])
            if kw != kernel_size_raw:
                kernel_size_raw = min(kernel_size_raw, kw)
        except Exception:
            pass
        stride = int(self.stride_init)
        dilation = int(self.dilation_init)
        
        # PyTorch: kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        # MLX: 完全相同
        kernel_size = (kernel_size_raw - 1) * dilation + 1  # effective kernel size with dilations
        
        # PyTorch: padding_total = kernel_size - stride
        # MLX: 完全相同
        padding_total = kernel_size - stride
        
        # PyTorch: extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        # MLX: 使用原始(MLX)格式计算padding
        extra_padding = self.get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        # 记录 SConv1d pad 输入（MLX, BT*C）
        try:
            from indextts.utils.cfm_debugger import log_cfm_stage
            log_cfm_stage(
                f"wavenet_l{self.debug_layer_index}_sconv_pad_input",
                mlx_data={
                    'x_bt_c': x,
                    'shape_bt_c': [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
                    'head_bt_c': x.flatten()[:3],
                    'tail_bt_c': x.flatten()[-3:],
                },
                layer=int(self.debug_layer_index) if self.debug_layer_index is not None else 0,
            )
        except Exception:
            pass
        # 在(批, 序列, 通道)格式下进行padding（与PyTorch一致的时间维padding）
        if self.causal:
            # Left padding for causal
            x_padded = self.mlx_pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # 非因果：与 PyTorch 等价
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x_padded = self.mlx_pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        # 精确裁剪到目标长度，确保与 PyTorch 一致
        target_len = seq_len + padding_total + extra_padding
        if int(x_padded.shape[1]) != int(target_len):
            x_padded = x_padded[:, :int(target_len), :]
        # 调试：记录 SConv1d 的 padding 结果与参数
        try:
            from indextts.utils.cfm_debugger import log_cfm_stage
            x_padded_bt_c = x_padded  # 已经是 (B,T,C)
            log_cfm_stage(
                f"wavenet_l{self.debug_layer_index}_sconv_pad",
                mlx_data={
                    'x_padded': x_padded,
                    'x_padded_bt_c': x_padded_bt_c,
                    'shape': [int(x_padded.shape[0]), int(x_padded.shape[1]), int(x_padded.shape[2])],
                    'orig_T': int(seq_len),
                    'kernel_size_init': int(self.kernel_size_init),
                    'conv_weight_dim1': int(self.conv.conv.weight.shape[1]),
                    'kernel_size_raw': int(kernel_size_raw),
                    'kernel_size': int(kernel_size),
                    'stride': int(stride),
                    'dilation': int(dilation),
                    'padding_total': int(padding_total),
                    'extra_padding': int(extra_padding),
                    'target_len': int(target_len),
                    'x_padded_len': int(x_padded.shape[1]),
                },
                layer=int(self.debug_layer_index) if self.debug_layer_index is not None else 0,
            )
        except Exception:
            pass
        
        # 直接在(批, 序列, 通道)格式上进行卷积（MLX Conv1d 支持）
        output = self.conv(x_padded)
    
        return output

def fused_add_tanh_sigmoid_multiply_mlx(input_a, input_b, n_channels):
    """
    MLX implementation of fused gated activation.
    
    Args:
        input_a: (batch, 2*n_channels, seq_len)
        input_b: (batch, 2*n_channels, seq_len)
        n_channels: int
    
    Returns:
        output: (batch, n_channels, seq_len)
    """
    # Note: MLX uses (batch, seq_len, channels) format
    # So we need to adapt
    
    in_act = input_a + input_b
    
    # Split channels
    t_act = mx.tanh(in_act[:, :, :n_channels])
    s_act = mx.sigmoid(in_act[:, :, n_channels:])
    
    return t_act * s_act


class MLXWaveNet(nn.Module):
    """
    MLX implementation of WaveNet.
    使用经过测试验证的SConv1d实现，完全复刻PyTorch行为
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        # Input layers (dilated convolutions with tested SConv1d)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            # 与 Torch 状态字典一致：所有层均使用 SConv1d，kernel_size 来自配置
            layer = MLXSConv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                dilation=dilation,
                bias=True,
                causal=False,
                pad_mode='reflect'
            )
            self.in_layers.append(layer)
        
        # Residual/skip layers
        self.res_skip_layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                kernel_size=1,
                padding=0,
                bias=True
            )
            init_conv_pytorch_compatible(layer)
            self.res_skip_layers.append(layer)
        
        # Conditioning layer (if using global conditioning)
        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                kernel_size=1,
                padding=0,
                bias=True
            )
            init_conv_pytorch_compatible(self.cond_layer)
        else:
            self.cond_layer = None
        
        # Dropout
        self.dropout = nn.Dropout(p_dropout)
        
        print(f">> MLX WaveNet initialized (with tested SConv1d):")
        print(f"   Layers: {n_layers}, Channels: {hidden_channels}")
        print(f"   Kernel: {kernel_size}, Dilation rate: {dilation_rate}")
        print(f"   Using tested SConv1d implementation")
    
    def __call__(self, x, x_mask, g=None):
        """
        x: (batch, seq_len, channels) - 纯 MLX 格式
        x_mask: (batch, 1, seq_len) - mask (True = keep)
        g: (batch, seq_len, gin_channels) - global conditioning
        
        Returns:
            output: (batch, seq_len, channels) - 纯 MLX 格式
        """
        # 纯 MLX 实现，无需格式转换
        x_mlx = x  # 直接使用 MLX 格式
        
        # 检查mask的形状并适当处理
        if len(x_mask.shape) == 3 and x_mask.shape[1] == 1:
            # 输入是 (batch, 1, seq_len)，需要转换为 (batch, seq_len, 1)
            x_mask_mlx = x_mask.transpose((0, 2, 1))  # (batch, seq_len, 1)
        else:
            # 输入已经是 (batch, seq_len, 1) 格式
            x_mask_mlx = x_mask
        
        output = mx.zeros_like(x_mlx)  # 使用 MLX 格式
        
        # Process global conditioning (match PyTorch: compute at time length=1, broadcast on add)
        if g is not None and self.cond_layer is not None:
            # Normalize to (batch, 1, gin_channels)
            if len(g.shape) == 4:  # (B,1,1,G)
                g = g.squeeze(1).squeeze(1)  # (B,G)
                g = g.reshape(g.shape[0], 1, -1)
            elif len(g.shape) == 3:
                if g.shape[1] == 1:          # (B,1,G)
                    pass
                elif g.shape[2] == 1:        # (B,G,1) -> (B,1,G)
                    g = g.transpose(0, 2, 1)
                else:                         # (B,T,G) -> take first time step to keep time=1
                    g = g[:, :1, :]
            elif len(g.shape) == 2:          # (B,G) -> (B,1,G)
                g = g.reshape(g.shape[0], 1, -1)

            # 1x1 conv over time=1 → (B,1,2*hidden*n_layers)
            g_cond = self.cond_layer(g)
        else:
            g_cond = None
        
        for i in range(self.n_layers):
            # 仅在第0层，记录进入 WaveNet 的原始输入与 mask，用于与 PyTorch 对齐
            if i == 0:
                try:
                    from indextts.utils.cfm_debugger import log_cfm_stage
                    def head_tail(a):
                        flat = a.reshape(-1)
                        return flat[:3], flat[-3:]
                    hx, tx = head_tail(x_mlx)
                    hm, tm = head_tail(x_mask_mlx)
                    log_cfm_stage(
                        "wavenet_l0_input_raw",
                        mlx_data={
                            'x_raw': x_mlx,
                            'x_raw_shape': (x_mlx.shape[0], x_mlx.shape[1], x_mlx.shape[2]),
                            'x_raw_head': hx,
                            'x_raw_tail': tx,
                            'x_mask': x_mask_mlx,
                            'x_mask_shape': (x_mask_mlx.shape[0], x_mask_mlx.shape[1], x_mask_mlx.shape[2] if len(x_mask_mlx.shape)==3 else 1),
                            'x_mask_head': hm,
                            'x_mask_tail': tm,
                        },
                        layer=0,
                    )
                except Exception:
                    pass
            # Apply mask and in-layer conv
            x_masked = x_mlx * x_mask_mlx
            if i == 0:
                # 运行时检查第0层是否为1x1卷积
                try:
                    l0 = self.in_layers[0]
                    kdim = None
                    if hasattr(l0, 'conv') and hasattr(l0.conv, 'weight'):
                        kdim = int(l0.conv.weight.shape[1])
                    elif hasattr(l0, 'conv') and hasattr(l0.conv, 'conv') and hasattr(l0.conv.conv, 'weight'):
                        kdim = int(l0.conv.conv.weight.shape[1])
                    from indextts.utils.cfm_debugger import log_cfm_stage
                    log_cfm_stage(
                        "wavenet_l0_kernel_check",
                        mlx_data={
                            'layer0_type': str(type(self.in_layers[0])),
                            'kernel_dim1': int(kdim) if kdim is not None else -1,
                        },
                        layer=0,
                    )
                except Exception:
                    pass
            # Tag layer index for SConv1d logging
            if hasattr(self.in_layers[i], 'debug_layer_index'):
                self.in_layers[i].debug_layer_index = i
            x_in = self.in_layers[i](x_masked)  # (batch, seq_len, 2*hidden_channels)
            if i == 0:
                try:
                    from indextts.utils.cfm_debugger import log_cfm_stage
                    def head_tail(a):
                        flat = a.reshape(-1)
                        return flat[:3], flat[-3:]
                    h0, t0 = head_tail(x_masked)
                    h1, t1 = head_tail(x_in)
                    log_cfm_stage(
                        "wavenet_l0_in",
                        mlx_data={
                            'x_masked': x_masked,
                            'x_masked_shape': (x_masked.shape[0], x_masked.shape[1], x_masked.shape[2]),
                            'x_masked_head': h0,
                            'x_masked_tail': t0,
                            'x_in': x_in,
                            'x_in_shape': (x_in.shape[0], x_in.shape[1], x_in.shape[2]),
                            'x_in_head': h1,
                            'x_in_tail': t1,
                        },
                        layer=0,
                    )
                except Exception:
                    pass
            
            # Add global conditioning
            if g_cond is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g_cond[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)
            
            # Fused gated activation
            acts = fused_add_tanh_sigmoid_multiply_mlx(x_in, g_l, self.hidden_channels)

            # Dropout (disabled to match PyTorch eval behavior)
            acts = acts
            if i == 0:
                try:
                    from indextts.utils.cfm_debugger import log_cfm_stage
                    def head_tail(a):
                        flat = a.reshape(-1)
                        return flat[:3], flat[-3:]
                    hg, tg = head_tail(g_l)
                    ha, ta = head_tail(acts)
                    log_cfm_stage(
                        "wavenet_l0_gate",
                        mlx_data={
                            'g_l': g_l,
                            'g_l_shape': (g_l.shape[0], g_l.shape[1], g_l.shape[2]),
                            'g_l_head': hg,
                            'g_l_tail': tg,
                            'acts': acts,
                            'acts_shape': (acts.shape[0], acts.shape[1], acts.shape[2]),
                            'acts_head': ha,
                            'acts_tail': ta,
                        },
                        layer=0,
                    )
                except Exception:
                    pass
            
            # Residual/skip connection
            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                skip_acts = res_skip_acts[:, :, self.hidden_channels:]
                # 期望等长；若不等长，记录日志但不裁剪（保持与 Torch 逻辑一致）
                if res_acts.shape[1] != x_mlx.shape[1]:
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage(
                            "wavenet_l0_length_mismatch",
                            mlx_data={
                                'x_len': int(x_mlx.shape[1]),
                                'res_len': int(res_acts.shape[1]),
                                'skip_len': int(skip_acts.shape[1]),
                                'x_in_len': int(x_in.shape[1]),
                                'acts_len': int(acts.shape[1]),
                            },
                            layer=0,
                        )
                    except Exception:
                        pass
                # 严格断言，快速暴露长度不一致来源
                assert res_acts.shape[1] == x_mlx.shape[1], (
                    f"WaveNet layer {i} length mismatch: x={int(x_mlx.shape[1])}, res={int(res_acts.shape[1])}, "
                    f"x_in={int(x_in.shape[1])}, acts={int(acts.shape[1])}"
                )
                
                x_mlx = (x_mlx + res_acts) * x_mask_mlx
                output = output + skip_acts
                if i == 0:
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        def head_tail(a):
                            flat = a.reshape(-1)
                            return flat[:3], flat[-3:]
                        hr, tr = head_tail(res_acts)
                        hs, ts = head_tail(skip_acts)
                        log_cfm_stage(
                            "wavenet_l0_res_skip",
                            mlx_data={
                                'res_acts': res_acts,
                                'res_acts_shape': (res_acts.shape[0], res_acts.shape[1], res_acts.shape[2]),
                                'res_acts_head': hr,
                                'res_acts_tail': tr,
                                'skip_acts': skip_acts,
                                'skip_acts_shape': (skip_acts.shape[0], skip_acts.shape[1], skip_acts.shape[2]),
                                'skip_acts_head': hs,
                                'skip_acts_tail': ts,
                            },
                            layer=0,
                        )
                    except Exception:
                        pass
            else:
                # Last layer: only skip
                if res_skip_acts.shape[1] != x_mlx.shape[1]:
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage(
                            "wavenet_last_length_mismatch",
                            mlx_data={
                                'x_len': int(x_mlx.shape[1]),
                                'last_len': int(res_skip_acts.shape[1]),
                            },
                            layer=self.n_layers-1,
                        )
                    except Exception:
                        pass
                output = output + res_skip_acts
        
        # 纯 MLX 实现，直接返回 MLX 格式
        return output * x_mask_mlx  # Apply mask in MLX format

