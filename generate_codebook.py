#!/usr/bin/env python3
"""Optimized codebook generation with automatic size determination."""

import os
import numpy as np
import torch
import pickle
import click
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import dnnlib
import legacy

#----------------------------------------------------------------------------

def analyze_model_dimensions(network_pkl, device='cuda'):
    """Analyze model dimensions to suggest codebook parameters."""
    print(f'Analyzing model dimensions from "{network_pkl}"...')
    
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    G.eval().requires_grad_(False)
    
    dimensions = {
        'z_dim': G.z_dim,
        'c_dim': G.c_dim,
        'img_resolution': G.img_resolution,
        'img_channels': G.img_channels,
    }
    
    # Check if it's w+ or w
    z = torch.randn([1, G.z_dim], device=device)
    label = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(z, label, truncation_psi=0.7)
    
    if w.dim() == 3:
        dimensions['w_plus'] = True
        dimensions['num_ws'] = w.shape[1]
        dimensions['w_dim'] = w.shape[2]
    else:
        dimensions['w_plus'] = False
        dimensions['w_dim'] = w.shape[1]
    
    return dimensions

def suggest_codebook_size(dimensions):
    """Suggest optimal codebook size based on model dimensions."""
    img_res = dimensions['img_resolution']
    w_plus = dimensions.get('w_plus', False)
    
    # Base suggestions
    suggestions = {
        'codebook_size': {
            256: 256,
            512: 512,
            1024: 1024,
            1280: 2048,
            1920: 4096,
        }.get(img_res, 1024),
        'num_samples': 20000,
        'use_pca': img_res >= 1024,
        'pca_components': min(256, dimensions['w_dim']),
    }
    
    # Adjust for w+ models (need more samples)
    if w_plus:
        suggestions['num_samples'] = 40000
        suggestions['codebook_size'] = min(2048, suggestions['codebook_size'] * 2)
    
    # Adjust for conditional models
    if dimensions['c_dim'] > 0:
        suggestions['codebook_size'] = int(suggestions['codebook_size'] * 1.5)
        suggestions['num_samples'] = 30000
    
    print(f"\nModel Analysis Results:")
    print(f"  Resolution: {dimensions['img_resolution']}x{dimensions['img_resolution']}")
    print(f"  z_dim: {dimensions['z_dim']}")
    print(f"  w_dim: {dimensions['w_dim']}")
    print(f"  c_dim: {dimensions['c_dim']}")
    print(f"  w+: {dimensions.get('w_plus', False)}")
    print(f"\nSuggested Parameters:")
    print(f"  Codebook size: {suggestions['codebook_size']}")
    print(f"  Number of samples: {suggestions['num_samples']}")
    print(f"  Use PCA: {suggestions['use_pca']}")
    print(f"  PCA components: {suggestions.get('pca_components', 'N/A')}")
    
    return suggestions

#----------------------------------------------------------------------------

def generate_codebook(network_pkl, codebook_size=1024, num_samples=20000, 
                     use_pca=False, pca_components=256, seed=0, device='cuda'):
    """Generate optimized codebook."""
    
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    G.eval().requires_grad_(False)
    
    # Collect w vectors with varied truncation
    print(f'Collecting {num_samples} w vectors...')
    w_vectors = []
    
    # Different sampling strategies
    sampling_strategies = [
        ('uniform', {'truncation_psi': 1.0, 'num_samples': num_samples // 3}),
        ('truncated', {'truncation_psi': 0.7, 'num_samples': num_samples // 3}),
        ('diverse', {'truncation_psi': lambda: np.random.uniform(0.5, 1.0), 
                     'num_samples': num_samples // 3}),
    ]
    
    total_collected = 0
    for strategy_name, strategy_params in sampling_strategies:
        strategy_samples = strategy_params['num_samples']
        print(f'  Strategy: {strategy_name} ({strategy_samples} samples)')
        
        with torch.no_grad():
            for i in tqdm(range(strategy_samples)):
                # Generate random latent
                z = torch.randn([1, G.z_dim], device=device)
                label = torch.zeros([1, G.c_dim], device=device)
                
                # Apply truncation
                if callable(strategy_params['truncation_psi']):
                    trunc_psi = strategy_params['truncation_psi']()
                else:
                    trunc_psi = strategy_params['truncation_psi']
                
                # Get w vector
                if hasattr(G, 'mapping'):
                    w = G.mapping(z, label, truncation_psi=trunc_psi)
                else:
                    w = z
                
                # Extract appropriate w vector
                if w.dim() == 3:  # w+ style
                    # For w+, we can use different strategies:
                    # 1. Use first layer (most semantic)
                    # 2. Use mean across layers
                    # 3. Use a specific layer
                    w_rep = w[:, 0, :]  # Option 1: first layer
                else:  # w style
                    w_rep = w
                
                w_vectors.append(w_rep.cpu().numpy())
                total_collected += 1
    
    w_vectors = np.concatenate(w_vectors, axis=0)
    print(f'\nCollected {total_collected} w vectors, shape: {w_vectors.shape}')
    
    # Apply PCA if requested (for high-dimensional or redundant data)
    pca = None
    if use_pca and w_vectors.shape[1] > pca_components:
        print(f'Applying PCA from {w_vectors.shape[1]} to {pca_components} dimensions...')
        pca = PCA(n_components=pca_components, random_state=seed)
        w_vectors = pca.fit_transform(w_vectors)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f'  Explained variance: {explained_variance:.3f}')
    
    # K-Means clustering with optimal initialization
    print(f'\nPerforming K-Means clustering with {codebook_size} clusters...')
    
    # Determine optimal n_init based on codebook size
    n_init = max(3, min(10, codebook_size // 100))
    
    kmeans = KMeans(
        n_clusters=codebook_size,
        init='k-means++',
        n_init=n_init,
        max_iter=300,
        random_state=seed,
        verbose=1,
        algorithm='elkan' if w_vectors.shape[0] > 10000 else 'full'
    )
    
    kmeans.fit(w_vectors)
    codebook = kmeans.cluster_centers_
    
    # Transform back if PCA was used
    if pca is not None:
        print('Transforming codebook back to original space...')
        codebook = pca.inverse_transform(codebook)
    
    print(f'\nCodebook generation complete:')
    print(f'  Final shape: {codebook.shape}')
    print(f'  Inertia: {kmeans.inertia_:.4f}')
    print(f'  Iterations: {kmeans.n_iter_}')
    
    # Calculate quality metrics
    if w_vectors.shape[0] > 0:
        # Average distance to nearest center
        distances = kmeans.transform(w_vectors)
        min_distances = distances.min(axis=1)
        avg_distance = min_distances.mean()
        std_distance = min_distances.std()
        
        print(f'  Avg distance to center: {avg_distance:.4f}')
        print(f'  Std distance to center: {std_distance:.4f}')
    
    return codebook, kmeans, pca

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--output', 'output_file', help='Output codebook filename (.npy)', required=True)
@click.option('--codebook-size', type=int, help='Size of codebook (if not specified, auto-detect)')
@click.option('--num-samples', type=int, help='Number of w vectors to sample')
@click.option('--seed', type=int, default=0, help='Random seed')
@click.option('--device', type=str, default='cuda', help='Device to use')
@click.option('--auto', is_flag=True, help='Auto-detect optimal parameters')
def generate_codebook_cmd(
    network_pkl: str,
    output_file: str,
    codebook_size: int,
    num_samples: int,
    seed: int,
    device: str,
    auto: bool
):
    """Generate optimized codebook for StyleGAN models."""
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Analyze model to get dimensions
    dimensions = analyze_model_dimensions(network_pkl, device)
    
    # Get suggested parameters
    suggestions = suggest_codebook_size(dimensions)
    
    # Use auto-detected parameters if auto flag is set
    if auto:
        if codebook_size is None:
            codebook_size = suggestions['codebook_size']
        if num_samples is None:
            num_samples = suggestions['num_samples']
        use_pca = suggestions['use_pca']
        pca_components = suggestions['pca_components']
        
        print(f'\nUsing auto-detected parameters:')
        print(f'  Codebook size: {codebook_size}')
        print(f'  Number of samples: {num_samples}')
        print(f'  Use PCA: {use_pca}')
        print(f'  PCA components: {pca_components}')
    else:
        # Use user-specified parameters with defaults
        if codebook_size is None:
            codebook_size = 1024
        if num_samples is None:
            num_samples = 20000
        use_pca = False  # Default to no PCA
        pca_components = min(256, dimensions['w_dim'])
    
    # Generate codebook
    codebook, kmeans, pca = generate_codebook(
        network_pkl=network_pkl,
        codebook_size=codebook_size,
        num_samples=num_samples,
        use_pca=use_pca,
        pca_components=pca_components,
        seed=seed,
        device=device
    )
    
    # Save codebook
    print(f'\nSaving codebook to "{output_file}"...')
    np.save(output_file, codebook)
    
    # Save metadata
    metadata = {
        'network': network_pkl,
        'model_dimensions': dimensions,
        'codebook_size': codebook_size,
        'num_samples': num_samples,
        'use_pca': use_pca,
        'pca_components': pca_components if pca is not None else None,
        'explained_variance': pca.explained_variance_ratio_.sum() if pca is not None else 1.0,
        'kmeans_inertia': kmeans.inertia_,
        'kmeans_iterations': kmeans.n_iter_,
        'seed': seed,
        'generated_date': np.datetime64('now'),
    }
    
    metadata_file = output_file.replace('.npy', '_metadata.json')
    import json
    with open(metadata_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.datetime64):
                return str(obj)
            else:
                return obj
        
        serializable_metadata = {k: convert(v) for k, v in metadata.items()}
        json.dump(serializable_metadata, f, indent=2)
    
    print(f'Saved metadata to "{metadata_file}"')
    
    # Save visualization if possible
    try:
        if codebook.shape[1] == 512:  # Only visualize for 512-dim codebooks
            visualize_codebook(codebook, output_file)
    except:
        print('Could not generate visualization')
    
    print('\nDone!')

#----------------------------------------------------------------------------

def visualize_codebook(codebook, output_file):
    """Create visualization of codebook distribution."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import umap
        
        print('Creating codebook visualization...')
        
        # Sample a subset for visualization
        n_samples = min(1000, codebook.shape[0])
        indices = np.random.choice(codebook.shape[0], n_samples, replace=False)
        sampled_codebook = codebook[indices]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        codebook_tsne = tsne.fit_transform(sampled_codebook)
        
        axes[0].scatter(codebook_tsne[:, 0], codebook_tsne[:, 1], alpha=0.6, s=10)
        axes[0].set_title('t-SNE Visualization')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        
        # 2. UMAP visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        codebook_umap = reducer.fit_transform(sampled_codebook)
        
        axes[1].scatter(codebook_umap[:, 0], codebook_umap[:, 1], alpha=0.6, s=10)
        axes[1].set_title('UMAP Visualization')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        
        # 3. Distance distribution
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(sampled_codebook[:100])  # Use first 100 for efficiency
        axes[2].hist(distances, bins=50, alpha=0.7)
        axes[2].set_title(f'Distance Distribution (mean={distances.mean():.2f})')
        axes[2].set_xlabel('Distance between codebook points')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        vis_file = output_file.replace('.npy', '_visualization.png')
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Saved visualization to "{vis_file}"')
        
    except ImportError:
        print('Visualization requires matplotlib, scikit-learn, and umap-learn')
    except Exception as e:
        print(f'Visualization failed: {e}')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_codebook_cmd()