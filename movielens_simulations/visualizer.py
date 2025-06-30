# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_coclustered_matrix(reordered_ratings, save_path=None, filename='clusters/coclustering_result.png'):
    """Visualize coclustered matrix."""
    plt.figure(figsize=(12, 12))
    plt.imshow(reordered_ratings, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Normalized Ratings')
    plt.title('Coclustered MovieLens Ratings')
    plt.xlabel('Movies (sorted by cluster)')
    plt.ylabel('Users (sorted by cluster)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()