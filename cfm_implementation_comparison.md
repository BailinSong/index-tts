# CFM PyTorch vs MLX 实现逐行对比

## 1. 初始化差异

### PyTorch CFM (flow_matching.py)
```python
@torch.inference_mode()
def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
    B, T = mu.size(0), mu.size(1)
    # 确保使用固定的随机种子
    torch.manual_seed(42)
    z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
    return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
```

### MLX CFM (mlx_cfm.py)
```python
def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
    # Convert to MLX if needed
    if isinstance(mu, torch.Tensor):
        from indextts.utils.mlx_utils import torch_to_mlx, mlx_to_torch
        mu_mlx = torch_to_mlx(mu.cpu())
        x_lens_mlx = torch_to_mlx(x_lens.cpu())
        prompt_mlx = torch_to_mlx(prompt.cpu())
        style_mlx = torch_to_mlx(style.cpu())
        convert_back = True
        device = mu.device
    else:
        mu_mlx = mu
        x_lens_mlx = x_lens
        prompt_mlx = prompt
        style_mlx = style
        convert_back = False
    
    batch, seq_len, _ = mu_mlx.shape
    
    # Initialize noise - 使用与PyTorch相同的随机种子
    import torch
    torch.manual_seed(42)  # 确保与PyTorch版本同步
    mx.random.seed(42)
    z = mx.random.normal((batch, self.in_channels, seq_len)) * temperature
    
    # Create time span
    t_span = mx.linspace(0, 1, n_timesteps + 1)
    
    # Solve ODE
    result = self.solve_euler(z, x_lens_mlx, prompt_mlx, mu_mlx, style_mlx, f0, t_span, inference_cfg_rate)
    
    # Convert back if needed
    if convert_back:
        result = mlx_to_torch(result, device=device)
    
    return result
```

## 2. 关键差异分析

### 差异1: 随机数生成
- **PyTorch**: `torch.randn([B, self.in_channels, T], device=mu.device)`
- **MLX**: `mx.random.normal((batch, self.in_channels, seq_len))`

**问题**: 即使设置了相同的种子，PyTorch 和 MLX 的随机数生成器可能产生不同的结果。

### 差异2: 数据类型转换
- **PyTorch**: 直接在 GPU 上操作
- **MLX**: 需要先转换到 CPU，再转换到 MLX，最后再转换回 PyTorch

**问题**: 多次转换可能导致精度损失。

### 差异3: 时间跨度生成
- **PyTorch**: `torch.linspace(0, 1, n_timesteps + 1, device=mu.device)`
- **MLX**: `mx.linspace(0, 1, n_timesteps + 1)`

**问题**: 设备不同可能导致数值差异。

## 3. solve_euler 方法对比

### PyTorch 版本
```python
def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
    t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
    sol = []
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0
    if self.zero_prompt_speech_token:
        mu[..., :prompt_len] = 0
    for step in tqdm(range(1, len(t_span))):
        dt = t_span[step] - t_span[step - 1]
        if inference_cfg_rate > 0:
            # CFG implementation
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
            
            stacked_dphi_dt = self.estimator(
                stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
            )
            
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        else:
            dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
        
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
        x[:, :, :prompt_len] = 0
    
    return sol[-1]
```

### MLX 版本
```python
def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
    prompt_len = prompt.shape[-1]
    
    # Prepare prompt
    prompt_x = mx.zeros_like(x)
    prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
    x[:, :, :prompt_len] = 0
    
    # Zero out prompt speech tokens if needed
    if self.zero_prompt_speech_token:
        mu[:, :prompt_len, :] = 0
    
    # 初始化时间变量
    t = t_span[0]
    
    # Euler iteration with progress
    num_steps = len(t_span) - 1
    print(f">> [MLX CFM] Starting Euler solver ({num_steps} steps)...")
    
    for step in range(1, len(t_span)):
        # 计算时间步长
        dt = t_span[step] - t_span[step - 1]
        
        if inference_cfg_rate > 0:
            # Classifier-free guidance: stack original and null inputs
            stacked_prompt_x = mx.concatenate([prompt_x, mx.zeros_like(prompt_x)], axis=0)
            stacked_style = mx.concatenate([style, mx.zeros_like(style)], axis=0)
            stacked_mu = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            stacked_x = mx.concatenate([x, x], axis=0)
            
            # Create timestep tensor for both batches
            t_scalar = mx.array([float(t)])
            stacked_t = mx.concatenate([t_scalar, t_scalar], axis=0)
            
            # Duplicate x_lens for both batches
            stacked_x_lens = mx.concatenate([x_lens, x_lens], axis=0)
            
            # Forward pass
            stacked_dphi_dt = self.estimator(
                stacked_x, stacked_prompt_x, stacked_x_lens, 
                stacked_t, stacked_style, stacked_mu,
                mask_content=False
            )
            
            # Split and apply CFG
            dphi_dt, cfg_dphi_dt = mx.split(stacked_dphi_dt, 2, axis=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        else:
            # No CFG
            t_scalar = mx.array([float(t)])
            dphi_dt = self.estimator(x, prompt_x, x_lens, t_scalar, style, mu)
        
        # Euler step
        x = x + dt * dphi_dt
        
        # 时间更新
        t = t + dt
        
        # Keep prompt unchanged
        x[:, :, :prompt_len] = 0
        
        # 计算下一步的dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
        
        # Force evaluation to avoid graph buildup
        mx.eval(x)
    
    return x
```

## 4. 关键差异总结

### 差异1: 时间变量处理
- **PyTorch**: `t.unsqueeze(0)` 创建形状为 (1,) 的张量
- **MLX**: `mx.array([float(t)])` 创建形状为 (1,) 的数组

### 差异2: 张量操作
- **PyTorch**: `torch.cat()` 和 `chunk()`
- **MLX**: `mx.concatenate()` 和 `mx.split()`

### 差异3: 随机数生成
- **PyTorch**: `torch.randn()` 在指定设备上生成
- **MLX**: `mx.random.normal()` 在 MLX 设备上生成

### 差异4: 数据类型转换
- **PyTorch**: 直接操作，无转换
- **MLX**: 需要多次转换 (PyTorch -> CPU -> MLX -> PyTorch)

## 5. 潜在问题

1. **随机数生成差异**: 即使设置相同种子，PyTorch 和 MLX 的随机数生成器可能产生不同结果
2. **精度损失**: 多次数据类型转换可能导致精度损失
3. **数值计算差异**: MLX 和 PyTorch 在某些数值计算上可能有微小差异
4. **设备差异**: 不同设备上的计算可能有微小差异

## 6. 建议修复方案

1. **统一随机数生成**: 使用相同的随机数生成策略
2. **减少转换**: 尽量减少数据类型转换
3. **数值精度**: 确保数值计算的精度一致性
4. **调试输出**: 添加详细的调试输出来对比中间结果
