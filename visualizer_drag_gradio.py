import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial
import json

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np

parser = ArgumentParser()
parser.add_argument('--share', action='store_true', default='True')
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
args = parser.parse_args()

cache_dir = args.cache_dir
device = 'cuda'

def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points

def clear_state(global_state, target=None):
    """Clear target history state from global_state."""
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state

def init_images(global_state):
    """Initialize images with advanced settings."""
    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    # Advanced codebook loading
    if 'codebook_path' in state and state['codebook_path'] is not None:
        if os.path.exists(state['codebook_path']):
            success = state['renderer'].load_codebook(
                state['codebook_path'],
                lambda_vq=state['params'].get('lambda_vq', 0.1),
                vq_mode=state['params'].get('vq_mode', 'static'),
                use_adaptive=state['params'].get('use_adaptive', False)
            )
            if success:
                print(f"Codebook loaded from {state['codebook_path']}")
                print(f"  Mode: {state['params'].get('vq_mode', 'static')}")
                print(f"  Lambda: {state['params'].get('lambda_vq', 0.1)}")
            else:
                print(f"Failed to load codebook from {state['codebook_path']}")
        else:
            print(f"Codebook path not found: {state['codebook_path']}")

    # Update renderer parameters
    renderer = state['renderer']
    renderer.vq_mode = state['params'].get('vq_mode', 'static')
    renderer.min_lambda = state['params'].get('min_lambda', 0.01)
    renderer.max_lambda = state['params'].get('max_lambda', 0.5)
    renderer.lambda_latent = state['params'].get('lambda_latent', 0.01)
    renderer.smoothing_window = state['params'].get('smoothing_window', 0)
    
    renderer.grad_projection_enabled = state['params'].get('grad_projection', True)
    renderer.pca_threshold = state['params'].get('pca_threshold', 0.95)
    
    renderer.tracking_mode = state['params'].get('tracking_mode', 'standard')
    renderer.patch_size = state['params'].get('patch_size', 5)
    renderer.spatial_penalty = state['params'].get('spatial_penalty', 0.1)

    state['renderer'].init_network(
        state['generator_params'],
        valid_checkpoints_dict[state['pretrained_weight']],
        state['params']['seed'],
        None,
        state['params']['latent_space'] == 'w+',
        'const',
        state['params']['trunc_psi'],
        state['params']['trunc_cutoff'],
        None,
        state['params']['lr']
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    
    # Clear statistics
    state['renderer'].stats_history = []
    
    try:
        if state['params'].get('grad_projection', True):
            state['renderer'].load_pca_components()
    except:
        pass
    
    return global_state

def update_image_draw(image, points, mask, show_mask, global_state=None):
    image_draw = draw_points_on_image(image, points)
    if global_state is not None and 'history_points' in global_state and len(global_state['history_points']) > 1:
        draw = ImageDraw.Draw(image_draw)
        history = global_state['history_points']
        
        for point_key in points.keys():
            line_coords = []
            for step_state in history:
                if point_key in step_state:
                    p_data = step_state[point_key]
                    curr_p = p_data.get("start_temp", p_data["start"])
                    line_coords.append((curr_p[1], curr_p[0]))
            
            if len(line_coords) > 1:
                draw.line(line_coords, fill='yellow', width=2)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw

def preprocess_mask_info(global_state, image):
    """Handle mask information."""
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    if (mask == 1).all():
        mask = last_mask

    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, do nothing to mask.')

    global_state['mask'] = updated_mask
    return global_state

valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

init_pkl = 'stylegan2_lions_512_pytorch'

with gr.Blocks() as app:
    global_state = gr.State({
        "images": {},
        "temporal_params": {},
        'mask': None,
        'last_mask': None,
        'show_mask': True,
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
            "lambda_vq": 0.1,
            "vq_mode": "adaptive",  # static, adaptive, multi-scale, hybrid
            "min_lambda": 0.01,
            "max_lambda": 0.5,
            "lambda_latent": 0.01,
            "smoothing_window": 3,
            "use_adaptive": True,
            "max_vq_distance": 50,
            "grad_projection": True,  
            "pca_threshold": 0.95,   
            "tracking_mode": "standard",  
            "patch_size": 5,  
            "spatial_penalty": 0.1,  
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl,
        'codebook_path': None,
        'codebook_status': "No codebook loaded",
        'stats_history': [],
        "history_points": [],
    })

    # init image
    global_state = init_images(global_state)

    with gr.Row():
        with gr.Row():
            # Left --> tools
            with gr.Column(scale=3):
                # Pickle
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Pickle', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(valid_checkpoints_dict.keys()),
                            label="Pretrained Model",
                            value=init_pkl,
                        )

                # Latent
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Latent', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        form_seed_number = gr.Number(
                            value=global_state.value['params']['seed'],
                            interactive=True,
                            label="Seed",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="Step Size")

                        with gr.Row():
                            with gr.Column(scale=2, min_width=10):
                                form_reset_image = gr.Button("Reset Image")
                            with gr.Column(scale=3, min_width=10):
                                form_latent_space = gr.Radio(
                                    ['w', 'w+'],
                                    value=global_state.value['params']
                                    ['latent_space'],
                                    interactive=True,
                                    label='Latent space to optimize',
                                    show_label=False,
                                )

                # Drag
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Drag', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button('Add Points')
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button('Reset Points')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(value=0,
                                                      label="Steps",
                                                      interactive=False)

                # Tracking Robustness (Option 1)
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Tracking', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Accordion("Tracking Robustness (Experiment)", open=True):
                            form_tracking_mode = gr.Radio(
                                choices=["standard", "robust"],
                                value=global_state.value['params']['tracking_mode'],
                                label="Tracking Algorithm",
                                info="Standard: Pixel-wise (Baseline) | Robust: Patch+Inertia (Ours)"
                            )
                            
                            with gr.Row():
                                form_patch_size = gr.Slider(
                                    minimum=1,
                                    maximum=9,
                                    value=global_state.value['params']['patch_size'],
                                    step=2,
                                    label="Patch Size (robust only)",
                                    info="Larger = more stable, slower"
                                )
                                
                                form_spatial_penalty = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.5,
                                    value=global_state.value['params']['spatial_penalty'],
                                    step=0.01,
                                    label="Spatial Penalty",
                                    info="Higher = less jumping"
                                )
                            
                            gr.Markdown("""
                            **Comparison Guide:**
                            - **Standard (Baseline)**: Original DragGAN algorithm. Fast but drifts in textureless areas.
                            - **Robust (Ours)**: Patch matching + spatial regularization. More stable in smooth regions.
                            
                            **For Fair Comparison:**
                            1. Set the same seed for both modes
                            2. Place points in textureless regions (e.g., lion's body)
                            3. Keep all other parameters identical
                            """)

                # Advanced VQ Constraints
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='VQ Constraints', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        # Codebook upload
                        form_codebook_upload = gr.File(
                            label="Upload Codebook (.npy)",
                            file_types=[".npy"],
                            interactive=True,
                            file_count="single"
                        )
                        
                        # VQ Mode selection
                        form_vq_mode = gr.Radio(
                            ['static', 'adaptive', 'multi-scale'],
                            value=global_state.value['params']['vq_mode'],
                            label="VQ Constraint Mode",
                            interactive=True
                        )
                        
                        # Base VQ weight
                        form_lambda_vq = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=global_state.value['params']['lambda_vq'],
                            step=0.01,
                            label="Base VQ Weight",
                            interactive=True
                        )
                        
                        # Adaptive parameters
                        with gr.Accordion("Adaptive Parameters", open=False):
                            with gr.Row():
                                form_min_lambda = gr.Slider(
                                    0.0, 0.1, 
                                    value=global_state.value['params']['min_lambda'],
                                    step=0.001, 
                                    label="Min Lambda"
                                )
                                form_max_lambda = gr.Slider(
                                    0.1, 1.0,
                                    value=global_state.value['params']['max_lambda'],
                                    step=0.01,
                                    label="Max Lambda"
                                )
                            form_max_vq_distance = gr.Slider(
                                10, 200,
                                value=global_state.value['params']['max_vq_distance'],
                                step=5,
                                label="Max VQ Distance"
                            )
                        
                        # Additional constraints
                        with gr.Accordion("Additional Constraints", open=False):
                            form_lambda_latent = gr.Slider(
                                0.0, 0.1,
                                value=global_state.value['params']['lambda_latent'],
                                step=0.001,
                                label="Latent L2 Weight"
                            )
                            
                            form_smoothing_window = gr.Slider(
                                0, 10,
                                value=global_state.value['params']['smoothing_window'],
                                step=1,
                                label="Trajectory Smoothing (0=off)"
                            )
                        
                        # Adaptive codebook
                        form_use_adaptive = gr.Checkbox(
                            label="Use Adaptive Codebook",
                            value=global_state.value['params']['use_adaptive'],
                            interactive=True
                        )
                        
                        # Codebook status
                        form_codebook_status = gr.Textbox(
                            label="Codebook Status",
                            value=global_state.value['codebook_status'],
                            interactive=False
                        )
                        
                        # Clear codebook button
                        form_clear_codebook = gr.Button("Clear Codebook", size="sm")
                        
                        # Statistics display
                        form_stats_display = gr.Textbox(
                            label="Current Statistics",
                            value="No statistics available",
                            visible=False,
                            interactive=False
                        )
                
                # PCA Gradient Projection
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='PCA Projection', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Accordion("PCA Gradient Projection", open=False):
                            form_grad_projection = gr.Checkbox(
                                label="Enable PCA Gradient Projection",
                                value=global_state.value['params'].get('grad_projection', True),
                                interactive=True
                            )
                            form_pca_threshold = gr.Slider(
                                0.5, 1.0,
                                value=global_state.value['params'].get('pca_threshold', 0.95),
                                step=0.01,
                                label="PCA Variance Threshold"
                            )
                            form_load_pca = gr.Button("Load PCA Components", size="sm")
                            form_pca_status = gr.Textbox(
                                label="PCA Status",
                                value="PCA not loaded",
                                interactive=False
                            )

                # Mask
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Mask', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        enable_add_mask = gr.Button('Edit Flexible Area')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]
                                ["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False)

            # Right --> Image
            with gr.Column(scale=8):
                form_image = ImageMask(
                    value=global_state.value['images']['image_show'],
                    brush_radius=20).style(
                        width=768,
                        height=768)

    # Documentation
    gr.Markdown("""
    ## Advanced Tracking Robustness Experiment (Option 1)
    
    ### Two Tracking Algorithms for Comparison:
    
    **1. Standard Mode (Baseline)**
    - Original DragGAN algorithm
    - Pixel-wise nearest neighbor matching
    - Fast but drifts in textureless regions
    
    **2. Robust Mode (Ours)**
    - Patch matching (5x5 feature patches)
    - Spatial regularization (prevents jumping)
    - More stable in smooth/textureless areas
    
    ### How to Conduct Fair Comparison:
    1. **Set up identical conditions**:
       - Same seed (for same initial image)
       - Same point locations
       - Same drag distance and direction
       - Same VQ/PCA settings (disable for pure tracking test)
    
    2. **Test in challenging scenarios**:
       - Place points in smooth regions (e.g., lion's body, sky, plain background)
       - Use medium to large drag distances
       - Observe point tracking stability
    
    3. **Compare results**:
       - Does the point follow correctly or drift?
       - Is the final deformation natural?
       - Check statistics for tracking confidence
    """)

    # Network & latents tab listeners
    def on_change_pretrained_dropdown(pretrained_value, global_state):
        global_state['pretrained_weight'] = pretrained_value
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state["images"]['image_show']

    form_pretrained_dropdown.change(
        on_change_pretrained_dropdown,
        inputs=[form_pretrained_dropdown, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_reset_image(global_state):
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state['images']['image_show']

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Update parameters
    def on_change_update_image_seed(seed, global_state):
        global_state["params"]["seed"] = int(seed)
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state['images']['image_show']

    form_seed_number.change(
        on_change_update_image_seed,
        inputs=[form_seed_number, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_latent_space(latent_space, global_state):
        global_state['params']['latent_space'] = latent_space
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state['images']['image_show']

    form_latent_space.change(on_click_latent_space,
                             inputs=[form_latent_space, global_state],
                             outputs=[global_state, form_image])

    # Tracking Mode Parameters
    def on_change_tracking_mode(tracking_mode, global_state):
        global_state['params']['tracking_mode'] = tracking_mode
        global_state['renderer'].tracking_mode = tracking_mode
        print(f'Tracking mode updated to: {tracking_mode}')
        return global_state

    form_tracking_mode.change(
        on_change_tracking_mode,
        inputs=[form_tracking_mode, global_state],
        outputs=[global_state]
    )
    
    def on_change_patch_size(patch_size, global_state):
        try:
            patch_size_int = int(float(patch_size))
        except (ValueError, TypeError):
            print(f"Ignoring invalid patch size: {patch_size}")
            return global_state
        
        global_state['params']['patch_size'] = patch_size_int
        global_state['renderer'].patch_size = patch_size_int
        print(f'Patch size updated to: {patch_size_int}')
        return global_state
    
    form_patch_size.change(
        on_change_patch_size,
        inputs=[form_patch_size, global_state],
        outputs=[global_state]
    )
    
    def on_change_spatial_penalty(spatial_penalty, global_state):
        try:
            spatial_penalty_float = float(spatial_penalty)
        except (ValueError, TypeError):
            print(f"Ignoring invalid spatial penalty: {spatial_penalty}")
            return global_state
        
        global_state['params']['spatial_penalty'] = spatial_penalty_float
        global_state['renderer'].spatial_penalty = spatial_penalty_float
        print(f'Spatial penalty updated to: {spatial_penalty_float}')
        return global_state
    
    form_spatial_penalty.change(
        on_change_spatial_penalty,
        inputs=[form_spatial_penalty, global_state],
        outputs=[global_state]
    )

    # VQ Parameters
    def on_change_vq_mode(vq_mode, global_state):
        global_state['params']['vq_mode'] = vq_mode
        global_state['renderer'].vq_mode = vq_mode
        print(f'VQ mode updated to: {vq_mode}')
        return global_state

    form_vq_mode.change(
        on_change_vq_mode,
        inputs=[form_vq_mode, global_state],
        outputs=[global_state]
    )
    
    def on_change_lambda_vq(lambda_vq, global_state):
        global_state['params']['lambda_vq'] = lambda_vq
        global_state['renderer'].lambda_vq = lambda_vq
        print(f'Base VQ weight updated to: {lambda_vq}')
        return global_state
    
    form_lambda_vq.change(
        on_change_lambda_vq,
        inputs=[form_lambda_vq, global_state],
        outputs=[global_state]
    )
    
    def on_change_min_lambda(min_lambda, global_state):
        global_state['params']['min_lambda'] = min_lambda
        global_state['renderer'].min_lambda = min_lambda
        print(f'Min lambda updated to: {min_lambda}')
        return global_state
    
    form_min_lambda.change(
        on_change_min_lambda,
        inputs=[form_min_lambda, global_state],
        outputs=[global_state]
    )
    
    def on_change_max_lambda(max_lambda, global_state):
        global_state['params']['max_lambda'] = max_lambda
        global_state['renderer'].max_lambda = max_lambda
        print(f'Max lambda updated to: {max_lambda}')
        return global_state
    
    form_max_lambda.change(
        on_change_max_lambda,
        inputs=[form_max_lambda, global_state],
        outputs=[global_state]
    )
    
    def on_change_max_vq_distance(max_vq_distance, global_state):
        global_state['params']['max_vq_distance'] = max_vq_distance
        global_state['renderer'].max_vq_distance = max_vq_distance
        print(f'Max VQ distance updated to: {max_vq_distance}')
        return global_state
    
    form_max_vq_distance.change(
        on_change_max_vq_distance,
        inputs=[form_max_vq_distance, global_state],
        outputs=[global_state]
    )
    
    def on_change_lambda_latent(lambda_latent, global_state):
        global_state['params']['lambda_latent'] = lambda_latent
        global_state['renderer'].lambda_latent = lambda_latent
        print(f'Latent L2 weight updated to: {lambda_latent}')
        return global_state
    
    form_lambda_latent.change(
        on_change_lambda_latent,
        inputs=[form_lambda_latent, global_state],
        outputs=[global_state]
    )
    
    def on_change_smoothing_window(smoothing_window, global_state):
        global_state['params']['smoothing_window'] = smoothing_window
        global_state['renderer'].smoothing_window = smoothing_window
        print(f'Smoothing window updated to: {smoothing_window}')
        return global_state
    
    form_smoothing_window.change(
        on_change_smoothing_window,
        inputs=[form_smoothing_window, global_state],
        outputs=[global_state]
    )
    
    def on_change_use_adaptive(use_adaptive, global_state):
        global_state['params']['use_adaptive'] = use_adaptive
        print(f'Use adaptive codebook: {use_adaptive}')
        return global_state
    
    form_use_adaptive.change(
        on_change_use_adaptive,
        inputs=[form_use_adaptive, global_state],
        outputs=[global_state]
    )
    
    # PCA Parameters
    def on_change_grad_projection(grad_projection, global_state):
        global_state['params']['grad_projection'] = grad_projection
        global_state['renderer'].grad_projection_enabled = grad_projection
        print(f'Gradient projection enabled: {grad_projection}')
        return global_state
    
    form_grad_projection.change(
        on_change_grad_projection,
        inputs=[form_grad_projection, global_state],
        outputs=[global_state]
    )
    
    def on_change_pca_threshold(pca_threshold, global_state):
        global_state['params']['pca_threshold'] = pca_threshold
        global_state['renderer'].pca_threshold = pca_threshold
        print(f'PCA threshold updated: {pca_threshold}')
        return global_state
    
    form_pca_threshold.change(
        on_change_pca_threshold,
        inputs=[form_pca_threshold, global_state],
        outputs=[global_state]
    )
    
    def on_load_pca(global_state):
        try:
            success = global_state['renderer'].load_pca_components()
            if success:
                status = "✓ PCA components loaded"
                print(status)
            else:
                status = "✗ PCA components not found"
                print(status)
            return global_state, status
        except Exception as e:
            status = f"✗ Error loading PCA: {str(e)}"
            print(status)
            return global_state, status
    
    form_load_pca.click(
        on_load_pca,
        inputs=[global_state],
        outputs=[global_state, form_pca_status]
    )

    # Codebook handling
    def on_upload_codebook(codebook_file, global_state):
        if codebook_file is not None:
            global_state['codebook_path'] = codebook_file.name
            global_state['codebook_status'] = f"Loading codebook from {os.path.basename(codebook_file.name)}..."
            
            try:
                success = global_state['renderer'].load_codebook(
                    codebook_file.name,
                    lambda_vq=global_state['params']['lambda_vq'],
                    vq_mode=global_state['params']['vq_mode'],
                    use_adaptive=global_state['params']['use_adaptive']
                )
                
                if success:
                    status = f"✓ Codebook loaded: {os.path.basename(codebook_file.name)}"
                    print(status)
                    global_state['codebook_status'] = status
                else:
                    status = "✗ Failed to load codebook"
                    print(status)
                    global_state['codebook_status'] = status
                    global_state['codebook_path'] = None
                    
            except Exception as e:
                status = f"✗ Error loading codebook: {str(e)}"
                print(status)
                global_state['codebook_status'] = status
                global_state['codebook_path'] = None
                
            return global_state, global_state['codebook_status']
        return global_state, global_state['codebook_status']

    form_codebook_upload.upload(
        on_upload_codebook,
        inputs=[form_codebook_upload, global_state],
        outputs=[global_state, form_codebook_status]
    )
    
    def on_clear_codebook(global_state):
        global_state['codebook_path'] = None
        global_state['codebook_status'] = "No codebook loaded"
        global_state['renderer'].codebook = None
        global_state['renderer'].adaptive_codebook = None
        global_state['renderer'].codebook_loaded = False
        print("Codebook cleared")
        return global_state, global_state['codebook_status']
    
    form_clear_codebook.click(
        on_clear_codebook,
        inputs=[global_state],
        outputs=[global_state, form_codebook_status]
    )

    # Drag functions
    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # Handle mask
        global_state = preprocess_mask_info(global_state, image)

        # Prepare points
        if len(global_state["points"]) == 0:
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (global_state, 0, global_state['images']['image_show'], 
                   form_reset_image, enable_add_points, enable_add_mask,
                   undo_points, form_reset_mask_btn, form_latent_space,
                   form_start_btn, form_stop_btn, form_pretrained_dropdown,
                   form_seed_number, form_lr_number, show_mask, form_lambda_number,
                   form_codebook_upload, form_vq_mode, form_lambda_vq,
                   form_min_lambda, form_max_lambda, form_max_vq_distance,
                   form_lambda_latent, form_smoothing_window, form_use_adaptive,
                   form_codebook_status, form_clear_codebook, 
                   form_grad_projection, form_pca_threshold, form_load_pca,
                   form_pca_status, form_tracking_mode, form_patch_size,
                   form_spatial_penalty,
                   gr.Textbox.update(value="No statistics available", visible=False))
        else:
            # Transform points
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state['mask']).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]
            global_state['temporal_params']['stop'] = False
            global_state['editing_state'] = 'running'

            # Reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            print(f'    Tracking Mode: {global_state["params"]["tracking_mode"]}')
            print(f'    VQ Mode: {global_state["params"]["vq_mode"]}')
            print(f'    PCA Projection: {global_state["params"]["grad_projection"]}')
            
            global_state["history_points"] = []
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break
                
                import copy
                current_points_snapshot = copy.deepcopy(global_state["points"])
                global_state["history_points"].append(current_points_snapshot)
                # Advanced drag with all constraints
                renderer._render_drag_impl(
                    global_state['generator_params'],
                    p_to_opt,
                    t_to_opt,
                    drag_mask,
                    global_state['params']['motion_lambda'],
                    reg=0,
                    feature_idx=5,
                    r1=global_state['params']['r1_in_pixels'],
                    r2=global_state['params']['r2_in_pixels'],
                    trunc_psi=global_state['params']['trunc_psi'],
                    is_drag=True,
                    to_pil=True,
                    lambda_vq=global_state['params']['lambda_vq'],
                    vq_mode=global_state['params']['vq_mode'],
                    min_lambda=global_state['params']['min_lambda'],
                    max_lambda=global_state['params']['max_lambda'],
                    lambda_latent=global_state['params']['lambda_latent'],
                    smoothing_window=global_state['params']['smoothing_window'],
                    tracking_mode=global_state['params']['tracking_mode'] 
                )

                if step_idx % global_state['draw_interval'] == 0:
                    # Update points
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt, t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [p_i[1], p_i[0]]
                        global_state["points"][key_point]["target"] = [t_i[1], t_i[0]]

                    # Update image
                    image_result = global_state['generator_params']['image']
                    image_draw = update_image_draw(
                        image_result,
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state,
                    )
                    global_state['images']['image_raw'] = image_result
                    
                    # Collect statistics
                    stats = global_state['generator_params'].get('stats', {})
                    stats_text = f"Step: {step_idx}\n"
                    stats_text += f"Tracking: {stats.get('tracking_mode', 'N/A')}\n"
                    stats_text += f"VQ Loss: {stats.get('vq_loss', 0):.4f}\n"
                    stats_text += f"Latent L2: {stats.get('latent_l2', 0):.4f}\n"
                    stats_text += f"Distance: {stats.get('distance', 0):.2f}\n"
                    stats_text += f"Lambda Effective: {stats.get('lambda_effective', 0):.3f}"
                    if 'grad_projection_ratio' in stats:
                        stats_text += f"\nGrad Projection: {stats.get('grad_projection_ratio', 0):.2f}"
                    if 'anchor_updated' in stats:
                        stats_text += f"\nAnchor Updated at step {stats.get('anchor_update_step', 0)}"

                yield (global_state, step_idx, global_state['images']['image_show'],
                       gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                       gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                       gr.Button.update(interactive=False), gr.Radio.update(interactive=False),
                       gr.Button.update(interactive=False), gr.Button.update(interactive=True),
                       gr.Dropdown.update(interactive=False), gr.Number.update(interactive=False),
                       gr.Number.update(interactive=False), gr.Button.update(interactive=False),
                       gr.Button.update(interactive=False), gr.Checkbox.update(interactive=False),
                       gr.Number.update(interactive=False), gr.File.update(interactive=False),
                       gr.Radio.update(interactive=False), gr.Slider.update(interactive=False),
                       gr.Slider.update(interactive=False), gr.Slider.update(interactive=False),
                       gr.Slider.update(interactive=False), gr.Slider.update(interactive=False),
                       gr.Slider.update(interactive=False), gr.Checkbox.update(interactive=False),
                       gr.Textbox.update(value=global_state['codebook_status']),
                       gr.Button.update(interactive=False),
                       gr.Checkbox.update(interactive=False),
                       gr.Slider.update(interactive=False),
                       gr.Button.update(interactive=False),
                       gr.Textbox.update(value=form_pca_status.value),
                       gr.Radio.update(interactive=False),
                       gr.Slider.update(interactive=False),
                       gr.Slider.update(interactive=False),
                       gr.Textbox.update(value=stats_text, visible=True))

                step_idx += 1

            # Final update
            image_result = global_state['generator_params']['image']
            global_state['images']['image_raw'] = image_result
            image_draw = update_image_draw(image_result, global_state['points'],
                                           global_state['mask'], global_state['show_mask'],
                                           global_state)

            global_state['editing_state'] = 'add_points'

            yield (global_state, 0, global_state['images']['image_show'],
                   gr.Button.update(interactive=True), gr.Button.update(interactive=True),
                   gr.Button.update(interactive=True), gr.Button.update(interactive=True),
                   gr.Button.update(interactive=True), gr.Radio.update(interactive=True),
                   gr.Button.update(interactive=True), gr.Button.update(interactive=False),
                   gr.Dropdown.update(interactive=True), gr.Number.update(interactive=True),
                   gr.Number.update(interactive=True), gr.Checkbox.update(interactive=True),
                   gr.Number.update(interactive=True), gr.File.update(interactive=True),
                   gr.Radio.update(interactive=True), gr.Slider.update(interactive=True),
                   gr.Slider.update(interactive=True), gr.Slider.update(interactive=True),
                   gr.Slider.update(interactive=True), gr.Slider.update(interactive=True),
                   gr.Slider.update(interactive=True), gr.Checkbox.update(interactive=True),
                   gr.Textbox.update(value=global_state['codebook_status']),
                   gr.Button.update(interactive=True),
                   gr.Checkbox.update(interactive=True),
                   gr.Slider.update(interactive=True),
                   gr.Button.update(interactive=True),
                   gr.Textbox.update(value=form_pca_status.value),
                   gr.Radio.update(interactive=True),
                   gr.Slider.update(interactive=True),
                   gr.Slider.update(interactive=True),
                   gr.Textbox.update(value="No statistics available", visible=False))

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state, form_steps_number, form_image,
            form_reset_image, enable_add_points, enable_add_mask,
            undo_points, form_reset_mask_btn, form_latent_space,
            form_start_btn, form_stop_btn, form_pretrained_dropdown,
            form_seed_number, form_lr_number, show_mask, form_lambda_number,
            form_codebook_upload, form_vq_mode, form_lambda_vq,
            form_min_lambda, form_max_lambda, form_max_vq_distance,
            form_lambda_latent, form_smoothing_window, form_use_adaptive,
            form_codebook_status, form_clear_codebook, form_grad_projection,
            form_pca_threshold, form_load_pca, form_pca_status,
            form_tracking_mode, form_patch_size, form_spatial_penalty,
            form_stats_display
        ],
    )

    def on_click_stop(global_state):
        global_state["temporal_params"]["stop"] = True
        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state, form_stop_btn])

    form_draw_interval_number.change(
        partial(
            on_change_single_global_state,
            "draw_interval",
            map_transform=lambda x: int(x),
        ),
        inputs=[form_draw_interval_number, global_state],
        outputs=[global_state],
    )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    # Mask
    def on_click_reset_mask(global_state):
        global_state['mask'] = np.ones(
            (
                global_state["images"]["image_raw"].size[1],
                global_state["images"]["image_raw"].size[0],
            ),
            dtype=np.uint8,
        )
        image_draw = update_image_draw(global_state['images']['image_raw'],
                                       global_state['points'],
                                       global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Image
    def on_click_enable_draw(global_state, image):
        """Function to start add mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to add_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_click_remove_draw(global_state, image):
        """Function to start remove mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to remove_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['edinting_state'] = 'remove_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    enable_add_mask.click(on_click_enable_draw,
                          inputs=[global_state, form_image],
                          outputs=[
                              global_state,
                              form_image,
                          ])

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """

        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'], mask,
                                       global_state['show_mask'], global_state)

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    enable_add_points.click(on_click_add_point,
                            inputs=[global_state, form_image],
                            outputs=[global_state, form_image])

    def on_click_image(global_state, evt: gr.SelectData):
        """This function only support click for point selection
        """
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                  'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_clear_points(global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        clear_state(global_state, target='point')

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, {}, global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    undo_points.click(on_click_clear_points,
                      inputs=[global_state],
                      outputs=[global_state, form_image])

    def on_click_show_mask(global_state, show_mask):
        """Function to control whether show mask on image."""
        global_state['show_mask'] = show_mask

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
    )

gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch(share=args.share, server_name="0.0.0.0" if args.listen else "127.0.0.1")