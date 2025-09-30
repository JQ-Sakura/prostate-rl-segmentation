import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from pathlib import Path

def plot_3d_comparison(volume, pred_mask, true_mask, case_name):
    """Generate 3D visualization comparison"""
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Original volume data
    ax1 = fig.add_subplot(131, projection='3d')
    plot_3d_volume(ax1, volume, title="Original Volume")
    
    # Predicted mask
    ax2 = fig.add_subplot(132, projection='3d')
    plot_3d_mask(ax2, pred_mask, title="Predicted Mask")
    
    # True mask
    ax3 = fig.add_subplot(133, projection='3d')
    plot_3d_mask(ax3, true_mask, title="Ground Truth")
    
    plt.suptitle(f"Case: {case_name}")
    plt.tight_layout()
    
    return fig

def plot_3d_volume(ax, volume, title):
    """Plot 3D volume data"""
    # Normalize the volume data
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Get the coordinates of non-zero voxels
    x, y, z = np.where(volume > 0.2)  # Set a threshold to reduce noise
    
    # Use the voxel values as colors
    colors = volume[x, y, z]
    
    # Plot the scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='gray',
                        alpha=0.1, marker='.')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_3d_mask(ax, mask, title):
    """Plot 3D mask"""
    # Get the non-zero points in the mask
    x, y, z = np.where(mask > 0)
    
    # Plot the scatter plot
    ax.scatter(x, y, z, c='red', alpha=0.1, marker='.')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def create_interactive_visualization(volume, pred_mask, true_mask, output_path):
    """Create interactive 3D visualization"""
    # Create figure
    fig = go.Figure()
    
    # Add original volume data
    x, y, z = np.where(volume > 0.2)
    colors = volume[x, y, z]
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            colorscale='Gray',
            opacity=0.1
        ),
        name='Original Volume'
    ))
    
    # Add predicted mask
    x, y, z = np.where(pred_mask > 0)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.3
        ),
        name='Predicted Mask'
    ))
    
    # Add true mask
    x, y, z = np.where(true_mask > 0)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.3
        ),
        name='Ground Truth'
    ))
    
    # Update layout
    fig.update_layout(
        title="3D Visualization of Cancer Region Detection",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=800
    )
    
    # Save as HTML file
    fig.write_html(output_path)

def create_slice_animation(volume, pred_mask, true_mask, output_path):
    """Create slice animation"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        
        # Display the original slice
        ax.imshow(volume[frame], cmap='gray')
        
        # Overlay the predicted mask
        pred_mask_slice = pred_mask[frame]
        ax.imshow(pred_mask_slice, alpha=0.3, cmap='Reds')
        
        # Overlay the true mask
        true_mask_slice = true_mask[frame]
        ax.imshow(true_mask_slice, alpha=0.3, cmap='Blues')
        
        ax.set_title(f'Slice {frame}')
        
    anim = FuncAnimation(fig, update, frames=len(volume),
                        interval=100, repeat=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow')
    plt.close()

def save_all_visualizations(volume, pred_mask, true_mask, case_name, output_dir):
    """Save all visualization results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save static 3D comparison figure
    fig = plot_3d_comparison(volume, pred_mask, true_mask, case_name)
    fig.savefig(output_dir / f"{case_name}_3d_comparison.png")
    plt.close(fig)
    
    # Save interactive 3D visualization
    create_interactive_visualization(
        volume, pred_mask, true_mask,
        output_dir / f"{case_name}_interactive.html"
    )
    
    # Save slice animation
    create_slice_animation(
        volume, pred_mask, true_mask,
        output_dir / f"{case_name}_slices.gif"
    ) 