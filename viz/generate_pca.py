"""Generate PCA components from codebook or latent samples."""

import numpy as np
import torch
import click
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import json

@click.command()
@click.option('--codebook', 'codebook_path', help='Codebook .npy file', required=True)
@click.option('--output', 'output_path', default='pca_components.npy', help='Output PCA components file')
@click.option('--n-components', type=int, default=256, help='Number of PCA components to keep')
@click.option('--variance-threshold', type=float, default=0.95, help='Variance threshold for auto component selection')
@click.option('--device', type=str, default='cuda', help='Device to use')
def generate_pca(codebook_path, output_path, n_components, variance_threshold, device):
    """Generate PCA components from codebook."""
    
    print(f'Loading codebook from {codebook_path}...')
    codebook = np.load(codebook_path)
    print(f'Codebook shape: {codebook.shape}')
    
    if codebook.dtype != np.float32:
        codebook = codebook.astype(np.float32)
    
    print('Computing PCA...')
    
    pca = PCA(n_components=min(n_components, codebook.shape[0]-1))
    pca.fit(codebook)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    n_components_auto = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components_auto = min(n_components_auto, n_components)
    
    print(f'Variance explained by {n_components} components: {cumulative_variance[-1]:.4f}')
    print(f'Components needed for {variance_threshold*100:.1f}% variance: {n_components_auto}')
    
    pca_final = PCA(n_components=n_components_auto)
    pca_final.fit(codebook)
    
    components = pca_final.components_  # [n_components, latent_dim]
    
    print(f'Final PCA components shape: {components.shape}')
    print(f'Explained variance ratio: {pca_final.explained_variance_ratio_.sum():.4f}')
    
    np.save(output_path, components)
    print(f'PCA components saved to {output_path}')
    
    metadata = {
        'source_codebook': codebook_path,
        'n_components': int(n_components_auto), 
        'explained_variance': float(pca_final.explained_variance_ratio_.sum()), 
        'components_shape': list(components.shape), 
        'variance_threshold': float(variance_threshold), 
    }
    
    metadata_path = output_path.replace('.npy', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'Metadata saved to {metadata_path}')
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
        axes[0].axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100:.0f}% variance')
        axes[0].axvline(x=n_components_auto, color='g', linestyle='--', label=f'{n_components_auto} components')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Explained Variance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        eigenvalues = pca_final.explained_variance_
        axes[1].bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7)
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Eigenvalue (Variance)')
        axes[1].set_title('PCA Eigenvalues')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        vis_file = output_path.replace('.npy', '_visualization.png')
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Visualization saved to {vis_file}')
        
    except ImportError:
        print('Visualization requires matplotlib (optional)')
    except Exception as e:
        print(f'Visualization failed: {e}')

if __name__ == '__main__':
    generate_pca()