# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

class AdaptiveCodebook:
    def __init__(self, initial_codebook, learning_rate=0.01, momentum=0.9, device='cuda'):
        self.codebook = initial_codebook.clone()
        self.momentum_buffer = torch.zeros_like(initial_codebook).to(device)
        self.lr = learning_rate
        self.momentum = momentum
        self.update_count = 0
        self.usage_counts = torch.zeros(initial_codebook.shape[0], device=device)  # 移动到正确设备
        self.device = device
        
    def update(self, w_samples, indices):
        """Update codebook using momentum."""
        if len(indices) == 0:
            return
        
        # 确保 indices 在正确设备上
        if indices.device != self.device:
            indices = indices.to(self.device)
        
        # Update usage counts
        unique_indices, counts = torch.unique(indices, return_counts=True)
        for idx, count in zip(unique_indices, counts):
            self.usage_counts[idx] += count
        
        # Update codebook with momentum
        for idx in unique_indices:
            mask = indices == idx
            if mask.sum() > 0:
                cluster_samples = w_samples[mask]
                centroid = cluster_samples.mean(dim=0)
                
                self.momentum_buffer[idx] = (
                    self.momentum * self.momentum_buffer[idx] + 
                    self.lr * (centroid - self.codebook[idx])
                )
                self.codebook[idx] += self.momentum_buffer[idx]
        
        self.update_count += 1
        
        # Occasionally add new codes in underutilized regions
        if self.update_count % 500 == 0:
            self._add_new_codes(w_samples)
    
    def _add_new_codes(self, w_samples):
        """Add new codes in underutilized regions."""
        if len(w_samples) == 0:
            return
        
        # Find least used codes
        n_to_replace = max(1, self.codebook.shape[0] // 100)
        _, least_used_indices = torch.topk(-self.usage_counts, n_to_replace)
        
        # Replace with random samples from current distribution
        for idx in least_used_indices:
            sample_idx = torch.randint(0, len(w_samples), (1,), device=self.device)
            self.codebook[idx] = w_samples[sample_idx]
            self.usage_counts[idx] = 0
            self.momentum_buffer[idx].zero_()

#----------------------------------------------------------------------------

class TrajectorySmoother:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.trajectory = []
        self.velocities = []
        
    def add_point(self, point):
        self.trajectory.append(point)
        if len(self.trajectory) > self.window_size:
            self.trajectory.pop(0)
        
        # Calculate velocity
        if len(self.trajectory) >= 2:
            vel = [
                self.trajectory[-1][0] - self.trajectory[-2][0],
                self.trajectory[-1][1] - self.trajectory[-2][1]
            ]
            self.velocities.append(vel)
            if len(self.velocities) > self.window_size - 1:
                self.velocities.pop(0)
    
    def get_smoothed(self):
        if len(self.trajectory) == 0:
            return None
        
        # Weighted moving average with velocity prediction
        weights = np.arange(1, len(self.trajectory) + 1)
        weights = weights / weights.sum()
        
        smoothed = np.zeros(2)
        for i, point in enumerate(self.trajectory):
            smoothed[0] += weights[i] * point[0]
            smoothed[1] += weights[i] * point[1]
        
        # Add velocity prediction if available
        if len(self.velocities) > 0:
            avg_velocity = np.mean(self.velocities, axis=0)
            smoothing_factor = 0.3
            smoothed[0] += smoothing_factor * avg_velocity[0]
            smoothed[1] += smoothing_factor * avg_velocity[1]
        
        return smoothed.tolist()
    
    def reset(self):
        self.trajectory = []
        self.velocities = []

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        
        # Advanced Codebook相关
        self.codebook = None
        self.adaptive_codebook = None
        self.codebook_loaded = False
        self.lambda_vq = 0.0  # VQ约束权重
        self.vq_mode = 'static'  # 'static', 'adaptive', 'multi-scale', 'hybrid'
        self.min_lambda = 0.01
        self.max_lambda = 0.5
        self.lambda_latent = 0.01  # 潜在空间L2约束权重
        self.max_vq_distance = 50  # 自适应权重最大距离
        
        # PCA Gradient Projection
        self.pca_components = None
        self.pca_loaded = False
        self.pca_threshold = 0.95
        self.grad_projection_enabled = True
        
        # Trajectory smoothing
        self.trajectory_smoothers = {}
        self.smoothing_window = 3
        
        # Multi-scale VQ weights
        self.layer_weights = None  # For w+ style
        
        # Statistics
        self.stats_history = []
        self.max_history_size = 100
        
        # Tracking Robustness (Option 1)
        self.tracking_mode = 'standard'  # 'standard' or 'robust'
        self.patch_size = 5  # For robust tracking
        self.spatial_penalty = 0.1  # Weight for spatial regularization
        
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()   # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def update_handle_points_patch(self, handle_points, feat_resize, feat_refs, h, w, r2, patch_size=3):
        new_points = []
        pad = patch_size // 2
        feat_padded = F.pad(feat_resize, (pad, pad, pad, pad), mode='replicate')
        
        for j, point in enumerate(handle_points):
            r = round(r2 / 512 * h)
            up = max(int(point[0]) - r, 0)
            down = min(int(point[0]) + r + 1, h)
            left = max(int(point[1]) - r, 0)
            right = min(int(point[1]) + r + 1, w)
            
            target_feat = feat_refs[j].reshape(1, -1, 1, 1) 

            best_dist = float('inf')
            best_p = point

            for py in range(up, down):
                for px in range(left, right):
                    curr_patch = feat_padded[:, :, py:py+patch_size, px:px+patch_size]
                    
                    dist = torch.linalg.norm(feat_resize[:, :, py, px].view(1,-1,1,1) - target_feat)
                    
                    spatial_dist = ((py - point[0])**2 + (px - point[1])**2)**0.5
                    weighted_dist = dist + 0.1 * spatial_dist 

                    if weighted_dist < best_dist:
                        best_dist = weighted_dist
                        best_p = [py, px]
            
            new_points.append(best_p)
        return new_points

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    # Advanced Codebook loading
    def load_codebook(self, codebook_path, lambda_vq=0.1, vq_mode='static', use_adaptive=False):
        """Load codebook with advanced options."""
        try:
            print(f'Loading codebook from "{codebook_path}"...')
            codebook_np = np.load(codebook_path)
            self.codebook = torch.from_numpy(codebook_np).to(self._device, dtype=self._dtype)
            
            if use_adaptive:
                self.adaptive_codebook = AdaptiveCodebook(self.codebook, device=self._device)
            
            self.lambda_vq = lambda_vq
            self.vq_mode = vq_mode
            self.codebook_loaded = True
            
            print(f'Codebook loaded: {codebook_np.shape}, mode: {vq_mode}, lambda_vq={lambda_vq}')
            
            # Initialize multi-scale weights for w+ if needed
            if 'multi-scale' in vq_mode:
                self._init_multi_scale_weights()
            
            return True
        except Exception as e:
            print(f'Failed to load codebook: {e}')
            self.codebook_loaded = False
            return False
    
    def load_pca_components(self, pca_path=None):
        """加载PCA主成分用于梯度投影"""
        try:
            import os
            if pca_path is None:
                possible_paths = [
                    './pca_components.npy',
                    './checkpoints/pca_components.npy',
                    os.path.join(os.path.dirname(__file__), 'pca_components.npy')
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pca_path = path
                        break
            
            if pca_path and os.path.exists(pca_path):
                pca_components = np.load(pca_path)
                
                if self._device.type == 'mps':
                    pca_dtype = torch.float32
                else:
                    pca_dtype = self._dtype
                
                self.pca_components = torch.from_numpy(pca_components).to(self._device, dtype=pca_dtype)
                self.pca_loaded = True
                print(f'PCA components loaded from {pca_path}, shape: {pca_components.shape}, dtype: {pca_dtype}')
                
                # 确保主成分矩阵方向正确
                if self.pca_components.dim() == 2:
                    if self.pca_components.shape[0] < self.pca_components.shape[1]:
                        k = min(self.pca_components.shape[0], 256)  
                        self.pca_components = self.pca_components[:k, :]
                    else:
                        k = min(self.pca_components.shape[1], 256)
                        self.pca_components = self.pca_components[:, :k].T
                
                print(f'Using {self.pca_components.shape[0]} PCA components for gradient projection')
                return True
            else:
                print('No PCA components found, gradient projection disabled')
                self.pca_loaded = False
                return False
                
        except Exception as e:
            print(f'Failed to load PCA components: {e}')
            self.pca_loaded = False
            return False

    def _init_multi_scale_weights(self):
        """Initialize layer weights for multi-scale VQ."""
        # Higher weight for early layers (more semantic), lower for later layers
        if hasattr(self, 'w') and self.w.dim() == 3:
            num_layers = self.w.shape[1]
            # Exponential decay weights
            self.layer_weights = torch.exp(-torch.arange(num_layers) / (num_layers / 3))
            self.layer_weights = self.layer_weights / self.layer_weights.sum()
            print(f'Initialized multi-scale weights: {self.layer_weights.tolist()}')
    
    def adaptive_vq_weight(self, current_distance, max_distance=None):
        """Compute adaptive VQ weight based on current drag distance."""
        if max_distance is None:
            max_distance = self.max_vq_distance
        
        # Normalize distance
        normalized_distance = min(current_distance / max_distance, 1.0)
        
        # Sigmoid function for smooth transition
        sigmoid = 1 / (1 + math.exp(-10 * (normalized_distance - 0.5)))
        lambda_vq = self.min_lambda + (self.max_lambda - self.min_lambda) * sigmoid
        
        return lambda_vq
    
    def compute_advanced_vq_loss(self, w_current, current_distance=None):
        """Compute advanced VQ loss with multiple strategies."""
        if not self.codebook_loaded or self.lambda_vq == 0:
            return torch.tensor(0.0, device=self._device, dtype=self._dtype)
        
        # Adaptive lambda
        if 'adaptive' in self.vq_mode and current_distance is not None:
            lambda_vq = self.adaptive_vq_weight(current_distance)
        else:
            lambda_vq = self.lambda_vq
        
        # Multi-scale VQ for w+ format
        if 'multi-scale' in self.vq_mode and w_current.dim() == 3:
            num_layers = w_current.shape[1]
            losses = []
            
            if self.layer_weights is None:
                self._init_multi_scale_weights()
            
            for i in range(num_layers):
                w_layer = w_current[:, i, :].to(self.codebook.dtype)
                layer_weight = self.layer_weights[i].to(self._device)
                layer_loss = self._compute_single_vq_loss(w_layer)
                losses.append(layer_weight * layer_loss)
            
            total_loss = torch.stack(losses).sum()
        else:
            # Single scale VQ
            w_flat = w_current[:, 0, :] if w_current.dim() == 3 else w_current
            w_flat = w_flat.to(self.codebook.dtype)
            total_loss = self._compute_single_vq_loss(w_flat)
        
        # Apply adaptive codebook update if enabled
        if self.adaptive_codebook is not None:
            w_flat = w_current[:, 0, :] if w_current.dim() == 3 else w_current
            w_flat = w_flat.to(self.codebook.dtype)
            distances = torch.cdist(w_flat, self.codebook)
            _, indices = torch.min(distances, dim=1)
            self.adaptive_codebook.update(w_flat, indices)
        
        return total_loss * lambda_vq
    
    def _compute_single_vq_loss(self, w_flat):
        """Compute single scale VQ loss."""
        if w_flat.dtype != self.codebook.dtype:
            w_flat = w_flat.to(self.codebook.dtype)
        
        distances = torch.cdist(w_flat, self.codebook)
        min_distances, _ = torch.min(distances, dim=1)
        
        # Use robust loss (less sensitive to outliers)
        vq_loss = torch.mean(torch.sqrt(min_distances**2 + 1e-6))
        return vq_loss

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.Adam([self.w], lr=lr)

        self.feat_refs = None
        self.points0_pt = None
        
        # Initialize trajectory smoothers
        self.trajectory_smoothers = {}
        
        # Store initial image for perceptual loss
        self.initial_img = None
        
        if not hasattr(self, 'pca_loaded') or not self.pca_loaded:
            self.load_pca_components()

    def update_lr(self, lr):
        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _render_drag_impl(self, res,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        lambda_vq       = None,
        vq_mode         = None,
        min_lambda      = None,
        max_lambda      = None,
        lambda_latent   = None,
        smoothing_window = None,
        tracking_mode   = 'standard',  
        **kwargs
    ):
        # Update parameters if provided
        if lambda_vq is not None:
            self.lambda_vq = lambda_vq
        if vq_mode is not None:
            self.vq_mode = vq_mode
        if min_lambda is not None:
            self.min_lambda = min_lambda
        if max_lambda is not None:
            self.max_lambda = max_lambda
        if lambda_latent is not None:
            self.lambda_latent = lambda_latent
        if smoothing_window is not None:
            self.smoothing_window = smoothing_window
        if tracking_mode is not None:
            self.tracking_mode = tracking_mode  # 设置追踪模式
        
        G = self.G
        ws = self.w
        
        # Ensure correct dtype
        ws = ws.to(self._dtype)
        
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        
        w0_typed = self.w0.to(self._dtype)
        ws = torch.cat([ws[:,:6,:], w0_typed[:,6:,:]], dim=1)
        
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
            self.trajectory_smoothers = {}
        self.points = points

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        
        ws_typed = ws.to(self._dtype)
        
        if is_drag:
            img, feat = G(ws_typed, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
        else:
            img = G(ws_typed, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True)
            feat = None

        h, w = G.img_resolution, G.img_resolution

        # Initialize loss_motion
        loss_motion = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        
        # Calculate current drag distance for adaptive weighting
        current_distance = 0
        if is_drag and len(points) > 0 and len(targets) > 0:
            # Use Euclidean distance of first point as reference
            current_distance = math.sqrt(
                (targets[0][0] - points[0][0])**2 + 
                (targets[0][1] - points[0][1])**2
            )
        
        if is_drag and feat is not None:
            X = torch.linspace(0, h, h, device=self._device, dtype=self._dtype)
            Y = torch.linspace(0, w, w, device=self._device, dtype=self._dtype)
            xx, yy = torch.meshgrid(X, Y, indexing='ij')
            feat_resize = F.interpolate(feat[feature_idx].to(self._dtype), [h, w], mode='bilinear')
            
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach().to(self._dtype), [h, w], mode='bilinear')
                self.feat_refs = []
                
                patch_size = self.patch_size if self.tracking_mode == 'robust' else 1
                pad = patch_size // 2
                
                if self.tracking_mode == 'robust':
                    feat0_padded = F.pad(self.feat0_resize, (pad, pad, pad, pad), mode='replicate')
                
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    if self.tracking_mode == 'robust':
                        ref_patch = feat0_padded[:, :, py:py+2*pad+1, px:px+2*pad+1]
                        self.feat_refs.append(ref_patch)
                    else:
                        # Standard 
                        self.feat_refs.append(self.feat0_resize[:,:,py,px])
                
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device).to(self._dtype) # 1, N, 2

            # Point tracking with feature matching 
            with torch.no_grad():
                for j, point in enumerate(points):
                    r = round(r2 / 512 * h)
                    up = max(point[0] - r, 0)
                    down = min(point[0] + r + 1, h)
                    left = max(point[1] - r, 0)
                    right = min(point[1] + r + 1, w)
                    
                    if self.tracking_mode == 'robust':
                        # === Robust Tracking Implementation (Ours) ===
                        patch_size = self.patch_size
                        pad = patch_size // 2
                        
                        feat_padded = F.pad(feat_resize, (pad, pad, pad, pad), mode='replicate')
                        target_patch = self.feat_refs[j]  # Shape: [1, C, patch_size, patch_size]
                        
                        best_dist = float('inf')
                        best_p = point
                        
                        for py_curr in range(up, down):
                            for px_curr in range(left, right):
                                curr_candidate = feat_padded[:, :, py_curr:py_curr+2*pad+1, px_curr:px_curr+2*pad+1]
                                
                                # Feature Distance (L2 between patches)
                                feat_dist = torch.linalg.norm(curr_candidate - target_patch)
                                
                                # Spatial Penalty 
                                spatial_dist = math.sqrt((py_curr - point[0])**2 + (px_curr - point[1])**2)
                                total_cost = feat_dist + self.spatial_penalty * spatial_dist
                                
                                if total_cost < best_dist:
                                    best_dist = total_cost
                                    best_p = [py_curr, px_curr]
                        
                        points[j] = best_p
                        
                    else:
                        # === Standard Tracking Implementation (Baseline) ===
                        feat_patch = feat_resize[:,:,up:down,left:right]
                        # self.feat_refs[j] is 1x1 pixel vector here
                        L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                        _, idx = torch.min(L2.view(1,-1), -1)
                        width = right - left
                        points[j] = [idx.item() // width + up, idx.item() % width + left]
                    
                    # Apply trajectory smoothing if enabled (Common to both)
                    if self.smoothing_window > 0:
                        if j not in self.trajectory_smoothers:
                            self.trajectory_smoothers[j] = TrajectorySmoother(self.smoothing_window)
                        self.trajectory_smoothers[j].add_point(points[j])
                        smoothed = self.trajectory_smoothers[j].get_smoothed()
                        if smoothed is not None:
                            points[j] = [int(smoothed[0]), int(smoothed[1])]

            res.points = [[point[0], point[1]] for point in points]

            # Motion supervision (Unchanged)
            loss_motion = torch.tensor(0.0, device=self._device, dtype=self._dtype)
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]]).to(self._device, dtype=self._dtype)
                if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx - point[0])**2 + (yy - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    feat_resize_typed = feat_resize.to(self._dtype)
                    target = F.grid_sample(feat_resize_typed, grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize_typed[:,:,relis,reljs].detach(), target)

        # Loss computation
        if is_drag:
            loss = loss_motion
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask.clone().detach().to(self._device)
                else:
                    mask_tensor = torch.tensor(mask, device=self._device).float()
                
                if mask_tensor.min() == 0 and mask_tensor.max() == 1:
                    mask_usq = mask_tensor.unsqueeze(0).unsqueeze(0).to(self._dtype)
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    loss += lambda_mask * loss_fix

            # Latent space L2 regularization
            latent_l2 = torch.mean((ws - w0_typed) ** 2)
            loss += self.lambda_latent * latent_l2
            
            # Advanced VQ constraint
            vq_loss = self.compute_advanced_vq_loss(ws, current_distance)
            loss += vq_loss
            
            # Collect statistics
            if 'stats' not in res:
                res.stats = {}
            res.stats['vq_loss'] = vq_loss.item() if hasattr(vq_loss, 'item') else vq_loss
            res.stats['latent_l2'] = latent_l2.item()
            res.stats['distance'] = current_distance
            res.stats['tracking_mode'] = self.tracking_mode
            res.stats['lambda_effective'] = self.lambda_vq if 'adaptive' not in self.vq_mode else self.adaptive_vq_weight(current_distance)
            
            # Store in history
            self.stats_history.append({
                'step': len(self.stats_history),
                'vq_loss': res.stats['vq_loss'],
                'distance': current_distance,
                'lambda': res.stats['lambda_effective'],
                'tracking_mode': self.tracking_mode
            })
            if len(self.stats_history) > self.max_history_size:
                self.stats_history.pop(0)
            
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                # PCA Gradient Projection
                if self.grad_projection_enabled and hasattr(self, 'pca_components') and self.pca_components is not None:
                    grad = self.w.grad  
                    
                    if grad is not None:
                        orig_shape = grad.shape
                        
                        # [batch, layers, dim] -> [batch*layers, dim]
                        if grad.dim() == 3:
                            batch_size, num_layers, dim = grad.shape
                            grad_flat = grad.reshape(-1, dim)
                        else:
                            grad_flat = grad.view(-1, grad.shape[-1])
                        
                        pca_components = self.pca_components.to(grad_flat.device).to(grad_flat.dtype)
                        
                        # grad_projected = (grad_flat @ pca_components.T) @ pca_components
                        # pca_components -> [k, dim]
                        proj_grad = torch.matmul(grad_flat, pca_components.T)
                        reconstructed_grad = torch.matmul(proj_grad, pca_components)
                        
                        residual_grad = grad_flat - reconstructed_grad
                        
                        exploration_factor = 0.1
                        final_grad = reconstructed_grad + exploration_factor * residual_grad
                        
                        self.w.grad = final_grad.view(orig_shape)
                        
                        if 'stats' in res:
                            projection_ratio = (reconstructed_grad.norm() / (grad_flat.norm() + 1e-8)).item()
                            res.stats['grad_projection_ratio'] = projection_ratio
                # -----------------------
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([self.w], 1.0)
                
                self.w_optim.step()
                
                # 动态锚点更新
                with torch.no_grad():
                    current_vq_loss = res.stats.get('vq_loss', float('inf'))
                    
                    if current_vq_loss < 0.1 and current_vq_loss > 0:
                        self.w0 = self.w.detach().clone()
                        
                        if hasattr(self, 'feat0_resize') and hasattr(self, 'feat_refs') and hasattr(self, 'points'):
                            self.feat_refs = []
                            patch_size = self.patch_size if self.tracking_mode == 'robust' else 1
                            pad = patch_size // 2
                            
                            if self.tracking_mode == 'robust':
                                feat0_padded = F.pad(self.feat0_resize, (pad, pad, pad, pad), mode='replicate')
                            
                            for point in self.points:
                                if len(point) >= 2:
                                    py, px = round(point[0]), round(point[1])
                                    if self.tracking_mode == 'robust':
                                        self.feat_refs.append(feat0_padded[:, :, py:py+2*pad+1, px:px+2*pad+1])
                                    else:
                                        self.feat_refs.append(self.feat0_resize[:,:,py,px])
                        
                        if 'stats' in res:
                            res.stats['anchor_updated'] = True
                            res.stats['anchor_update_step'] = len(self.stats_history)
                        
                        print(f"Anchor updated at step {len(self.stats_history)}, vq_loss: {current_vq_loss:.4f}")
        else:
            # Non-drag mode
            res.stop = True

        # Scale and convert to uint8.
        img = img[0] if img.dim() == 4 else img
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu().numpy()

#----------------------------------------------------------------------------