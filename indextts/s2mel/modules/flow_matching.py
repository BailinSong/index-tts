from abc import ABC

import torch
import torch.nn.functional as F

from indextts.s2mel.modules.diffusion_transformer import DiT
from indextts.s2mel.modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False
        
        # CFM 缓存相关
        self._cfm_cache_enabled = True
        self._cfm_cache_dir = "cfm_production_cache"

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, unified_random=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0: None
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        # 使用统一随机数生成器
        if unified_random is not None:
            z = unified_random.generate_noise((B, self.in_channels, T), device=mu.device) * temperature
        else:
            # 回退到原始方法
            torch.manual_seed(42)
            z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate, debug_layers=False)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5, debug_layers=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
        """
        # 缓存 CFM 输入数据
        self._cache_cfm_inputs(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
        
        # 简洁的输入信息（仅在debug模式下显示）
        if debug_layers:
            print(f"   x_lens: {x_lens}")
            print(f"   prompt: {prompt.shape}, min={prompt.min():.6f}, max={prompt.max():.6f}")
            print(f"   mu: {mu.shape}, min={mu.min():.6f}, max={mu.max():.6f}")
            print(f"   style: {style.shape}, min={style.min():.6f}, max={style.max():.6f}")
            print(f"   t_span: {t_span.shape}, min={t_span.min():.6f}, max={t_span.max():.6f}")
            print(f"   inference_cfg_rate: {inference_cfg_rate}")
        
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            
            # 缓存每步输入
            self._cache_cfm_step_input(step, x, t, prompt_x, style, mu, x_lens)
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                # 逐层调试：记录estimator输入
                if debug_layers:
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage("estimator_input", 
                                    pytorch_data={
                                        'x': stacked_x,
                                        'prompt_x': stacked_prompt_x, 
                                        'x_lens': x_lens,
                                        't': stacked_t,
                                        'style': stacked_style,
                                        'mu': stacked_mu
                                    },
                                    step=step,
                                    additional_info={'cfg_enabled': True})
                    except ImportError:
                        pass

                # 设置调试标志
                if debug_layers:
                    self.estimator._debug_layers = True

                # Duplicate x_lens for both batches (与MLX版本保持一致)
                stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)
                
                # 缓存 DiT 估计器输入
                self._cache_dit_input(step, stacked_x, stacked_prompt_x, stacked_x_lens, stacked_t, stacked_style, stacked_mu)
                
                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, stacked_x_lens, stacked_t, stacked_style, stacked_mu,
                )
                
                # 缓存 DiT 估计器输出
                self._cache_dit_output(step, stacked_dphi_dt)

                # 逐层调试：记录estimator输出（仅在debug模式下显示）
                if debug_layers:
                    print(f"   stacked_dphi_dt: {stacked_dphi_dt.shape}, min={stacked_dphi_dt.min():.6f}, max={stacked_dphi_dt.max():.6f}")
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage("estimator_output",
                                    pytorch_data={'dphi_dt': stacked_dphi_dt},
                                    step=step,
                                    additional_info={'cfg_enabled': True})
                    except ImportError:
                        pass

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                # 调试输出已移除
                if debug_layers:
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage("estimator_input", 
                                    pytorch_data={
                                        'x': x,
                                        'prompt_x': prompt_x, 
                                        'x_lens': x_lens,
                                        't': t.unsqueeze(0),
                                        'style': style,
                                        'mu': mu
                                    },
                                    step=step,
                                    additional_info={'cfg_enabled': False})
                    except ImportError:
                        pass
                
                # 设置调试标志
                if debug_layers:
                    self.estimator._debug_layers = True
                
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                
                # 逐层调试：记录estimator输出（仅在debug模式下显示）
                if debug_layers:
                    print(f"   dphi_dt: {dphi_dt.shape}, min={dphi_dt.min():.6f}, max={dphi_dt.max():.6f}")
                    try:
                        from indextts.utils.cfm_debugger import log_cfm_stage
                        log_cfm_stage("estimator_output",
                                    pytorch_data={'dphi_dt': dphi_dt},
                                    step=step,
                                    additional_info={'cfg_enabled': False})
                    except ImportError:
                        pass

            # 调试输出已移除

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            
            # 缓存每步输出
            self._cache_cfm_step_output(step, x, dphi_dt)
            
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        # 缓存 CFM 输出数据
        self._cache_cfm_output(sol[-1])
        
        return sol[-1]
    def forward(self, x1, x_lens, prompt_lens, mu, style, unified_random=None):
        """Computes diffusion loss

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x1: mel
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # 使用统一随机数生成器
        if unified_random is not None:
            # random timestep
            t = unified_random.generate_uniform((b, 1, 1), device=mu.device, dtype=x1.dtype)
            # sample noise p(x_0)
            z = unified_random.generate_noise_like(x1)
        else:
            # random timestep
            t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
            # sample noise p(x_0)
            z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z

    def _cache_cfm_inputs(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate):
        """缓存 CFM 输入数据"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import os
            import pickle
            import time
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'x': x,
                'x_lens': x_lens,
                'prompt': prompt,
                'mu': mu,
                'style': style,
                'f0': f0,
                't_span': t_span,
                'inference_cfg_rate': inference_cfg_rate,
                'timestamp': time.time(),
                'model_type': 'pytorch'
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_inputs_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"🔍 Cached PyTorch CFM inputs: {filename}")
            print(f"   📊 Shapes: x={x.shape}, mu={mu.shape}, prompt={prompt.shape}, style={style.shape}")
            print(f"   📊 x_lens: {x_lens.item()}, cfg_rate: {inference_cfg_rate}")
            
        except Exception as e:
            print(f"⚠️  CFM input caching failed: {e}")
    
    def _cache_cfm_output(self, output):
        """缓存 CFM 输出数据"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import os
            import pickle
            import time
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'output': output,
                'timestamp': time.time(),
                'model_type': 'pytorch'
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_output_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"🔍 Cached PyTorch CFM output: {filename}")
            print(f"   📊 Output shape: {output.shape}")
            
        except Exception as e:
            print(f"⚠️  CFM output caching failed: {e}")
    
    def _cache_cfm_step_input(self, step: int, x, t, prompt_x, style, mu, x_lens):
        """缓存 CFM 每步输入"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import time
            import os
            import pickle
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'step': step,
                'x': x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                't': t.detach().cpu() if isinstance(t, torch.Tensor) else t,
                'prompt_x': prompt_x.detach().cpu() if isinstance(prompt_x, torch.Tensor) else prompt_x,
                'style': style.detach().cpu() if isinstance(style, torch.Tensor) else style,
                'mu': mu.detach().cpu() if isinstance(mu, torch.Tensor) else mu,
                'x_lens': x_lens.detach().cpu() if isinstance(x_lens, torch.Tensor) else x_lens,
                'timestamp': time.time()
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_step_{step:03d}_input_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 详细输出
            x_stats = self._get_tensor_stats(x)
            t_stats = self._get_tensor_stats(t)
            mu_stats = self._get_tensor_stats(mu)
            style_stats = self._get_tensor_stats(style)
            prompt_stats = self._get_tensor_stats(prompt_x)
            
            print(f"🔍 PyTorch Step {step:03d} input:")
            print(f"   x: {x.shape}, min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, avg={x_stats['avg']:.6f}")
            print(f"   t: {t.shape}, min={t_stats['min']:.6f}, max={t_stats['max']:.6f}, avg={t_stats['avg']:.6f}")
            print(f"   mu: {mu.shape}, min={mu_stats['min']:.6f}, max={mu_stats['max']:.6f}, avg={mu_stats['avg']:.6f}")
            print(f"   style: {style.shape}, min={style_stats['min']:.6f}, max={style_stats['max']:.6f}, avg={style_stats['avg']:.6f}")
            print(f"   prompt: {prompt_x.shape}, min={prompt_stats['min']:.6f}, max={prompt_stats['max']:.6f}, avg={prompt_stats['avg']:.6f}")
            
        except Exception as e:
            print(f"⚠️  CFM step {step} input caching failed: {e}")
    
    def _cache_cfm_step_output(self, step: int, x, dphi_dt):
        """缓存 CFM 每步输出"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import time
            import os
            import pickle
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'step': step,
                'x': x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                'dphi_dt': dphi_dt.detach().cpu() if isinstance(dphi_dt, torch.Tensor) else dphi_dt,
                'timestamp': time.time()
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_step_{step:03d}_output_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 简化输出
            x_stats = self._get_tensor_stats(x)
            dphi_stats = self._get_tensor_stats(dphi_dt)
            print(f"🔍 PyTorch Step {step:03d} output: x={x.shape}, min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, avg={x_stats['avg']:.6f}")
            print(f"   dphi_dt={dphi_dt.shape}, min={dphi_stats['min']:.6f}, max={dphi_stats['max']:.6f}, avg={dphi_stats['avg']:.6f}")
            
        except Exception as e:
            print(f"⚠️  CFM step {step} output caching failed: {e}")
    
    def _cache_dit_input(self, step: int, x, prompt_x, x_lens, t, style, mu):
        """缓存 DiT 估计器输入"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import time
            import os
            import pickle
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'step': step,
                'process': 'dit_input',
                'x': x.detach().cpu() if isinstance(x, torch.Tensor) else x,
                'prompt_x': prompt_x.detach().cpu() if isinstance(prompt_x, torch.Tensor) else prompt_x,
                'x_lens': x_lens.detach().cpu() if isinstance(x_lens, torch.Tensor) else x_lens,
                't': t.detach().cpu() if isinstance(t, torch.Tensor) else t,
                'style': style.detach().cpu() if isinstance(style, torch.Tensor) else style,
                'mu': mu.detach().cpu() if isinstance(mu, torch.Tensor) else mu,
                'timestamp': time.time()
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_step_{step:03d}_dit_input_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 详细输出
            x_stats = self._get_tensor_stats(x)
            t_stats = self._get_tensor_stats(t)
            mu_stats = self._get_tensor_stats(mu)
            style_stats = self._get_tensor_stats(style)
            prompt_stats = self._get_tensor_stats(prompt_x)
            
            print(f"🔍 PyTorch Step {step:03d} DiT input:")
            print(f"   x: {x.shape}, min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, avg={x_stats['avg']:.6f}")
            print(f"   t: {t.shape}, min={t_stats['min']:.6f}, max={t_stats['max']:.6f}, avg={t_stats['avg']:.6f}")
            print(f"   mu: {mu.shape}, min={mu_stats['min']:.6f}, max={mu_stats['max']:.6f}, avg={mu_stats['avg']:.6f}")
            print(f"   style: {style.shape}, min={style_stats['min']:.6f}, max={style_stats['max']:.6f}, avg={style_stats['avg']:.6f}")
            print(f"   prompt: {prompt_x.shape}, min={prompt_stats['min']:.6f}, max={prompt_stats['max']:.6f}, avg={prompt_stats['avg']:.6f}")
            
        except Exception as e:
            print(f"⚠️  CFM step {step} DiT input caching failed: {e}")
    
    def _cache_dit_output(self, step: int, dphi_dt):
        """缓存 DiT 估计器输出"""
        if not self._cfm_cache_enabled:
            return
        
        try:
            import time
            import os
            import pickle
            
            # 创建缓存目录
            os.makedirs(self._cfm_cache_dir, exist_ok=True)
            
            # 准备缓存数据
            cache_data = {
                'step': step,
                'process': 'dit_output',
                'dphi_dt': dphi_dt.detach().cpu() if isinstance(dphi_dt, torch.Tensor) else dphi_dt,
                'timestamp': time.time()
            }
            
            # 保存缓存
            filename = f"cfm_pytorch_step_{step:03d}_dit_output_{int(time.time() * 1000)}.pkl"
            filepath = os.path.join(self._cfm_cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 详细输出
            dphi_stats = self._get_tensor_stats(dphi_dt)
            print(f"🔍 PyTorch Step {step:03d} DiT output:")
            print(f"   dphi_dt: {dphi_dt.shape}, min={dphi_stats['min']:.6f}, max={dphi_stats['max']:.6f}, avg={dphi_stats['avg']:.6f}")
            
        except Exception as e:
            print(f"⚠️  CFM step {step} DiT output caching failed: {e}")
    
    def _get_tensor_stats(self, tensor):
        """获取张量统计信息"""
        if tensor is None:
            return {"min": 0, "max": 0, "avg": 0}
        
        if isinstance(tensor, torch.Tensor):
            return {
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "avg": float(tensor.mean().item())
            }
        else:
            return {"min": 0, "max": 0, "avg": 0}


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
