import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr

from config import N_FOLDS, FIGURES_DIR
from geometry import (
    SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH, N_ZONES, ZONE_NAMES,
    D_LEFT_END, D_RIGHT_START,
)


def _save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'), dpi=150, bbox_inches='tight')


def _make_heatmap(ax, positions, values, cmap, title, nbins=20):
    """Helper to bin values spatially and show as heatmap."""
    x_edges = np.linspace(0, 1, nbins + 1)
    y_edges = np.linspace(0, 1, nbins + 1)
    val_map = np.full((nbins, nbins), np.nan)
    count_map = np.zeros((nbins, nbins))
    for i in range(len(positions)):
        xi = np.clip(np.searchsorted(x_edges, positions[i, 0]) - 1, 0, nbins - 1)
        yi = np.clip(np.searchsorted(y_edges, positions[i, 1]) - 1, 0, nbins - 1)
        if np.isnan(val_map[yi, xi]):
            val_map[yi, xi] = 0
        val_map[yi, xi] += values[i]
        count_map[yi, xi] += 1
    mean_map = np.where(count_map > 0, val_map / count_map, np.nan)
    im = ax.imshow(mean_map, origin='lower', aspect='equal', cmap=cmap, extent=[0, 1, 0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return im


def plot_umaze_geometry(positions, curvilinear_d, zone_labels):
    """Plot U-maze skeleton, curvilinear distance heatmap, and 3-zone classification."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # 1. Skeleton
    axes[0].scatter(positions[:, 0], positions[:, 1], c='lightgray', s=1, alpha=0.3)
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[0].plot([x1, x2], [y1, y2], 'r-', linewidth=3)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('U-maze skeleton on positions')
    axes[0].set_aspect('equal')

    # 2. Curvilinear distance colored
    sc = axes[1].scatter(positions[:, 0], positions[:, 1], c=curvilinear_d, s=1, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=axes[1], label='d (normalized curvilinear distance)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Curvilinear distance d along U-maze')
    axes[1].set_aspect('equal')

    # 3. Zone classification
    zone_colors = ['blue', 'green', 'red']
    for z in range(N_ZONES):
        mask_z = zone_labels == z
        axes[2].scatter(positions[mask_z, 0], positions[mask_z, 1],
                        c=zone_colors[z], s=1, alpha=0.3, label=ZONE_NAMES[z])
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('3-zone classification')
    axes[2].legend(markerscale=10)
    axes[2].set_aspect('equal')

    plt.tight_layout()
    _save_fig(fig, '01_umaze_geometry')
    plt.show()

    # Histogram of d with thresholds
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(curvilinear_d, bins=100, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=D_LEFT_END, color='blue', linestyle='--', linewidth=2,
               label=f'Left/Top ({D_LEFT_END:.3f})')
    ax.axvline(x=D_RIGHT_START, color='red', linestyle='--', linewidth=2,
               label=f'Top/Right ({D_RIGHT_START:.3f})')
    ax.set_xlabel('d (curvilinear distance)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of d + zone thresholds')
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, '01_d_histogram')
    plt.show()


def plot_training_curves(all_train_losses, all_val_losses):
    """Plot total loss per fold (train and validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))

    for fold in range(N_FOLDS):
        axes[0].plot(all_train_losses[fold], color=colors[fold], linewidth=1.5,
                     label=f'Fold {fold + 1}')
        axes[1].plot(all_val_losses[fold], color=colors[fold], linewidth=1.5,
                     label=f'Fold {fold + 1}')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total loss')
    axes[0].set_title('Train loss per fold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total loss')
    axes[1].set_title('Validation loss per fold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, '02_training_curves')
    plt.show()


def plot_loss_decomposition(all_train_losses_detail, all_val_losses_detail):
    """Plot decomposition of sub-losses across epochs (averaged over folds)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))
    loss_keys = ['cls', 'pos', 'd', 'feas']
    loss_titles = [
        'CrossEntropy (classification)',
        'GaussianNLL (regression)',
        'MSE (curvilinear distance d)',
        'Feasibility (outside maze)',
    ]

    for ax_idx, (key, title) in enumerate(zip(loss_keys, loss_titles)):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for fold in range(N_FOLDS):
            train_vals = [d[key] for d in all_train_losses_detail[fold]]
            val_vals = [d[key] for d in all_val_losses_detail[fold]]
            ax.plot(train_vals, color=colors[fold], linewidth=1, alpha=0.5)
            ax.plot(val_vals, color=colors[fold], linewidth=1.5, linestyle='--')
        ax.plot([], [], 'k-', linewidth=1, label='Train')
        ax.plot([], [], 'k--', linewidth=1.5, label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, '03_loss_decomposition')
    plt.show()


def plot_scatter_predictions(results, metrics):
    """Scatter plot: predicted vs true for X, Y, and curvilinear d."""
    y_test = results['y_test']
    y_pred = results['y_pred']
    d_test = results['d_test']
    d_pred = results['d_pred']
    r2_x = metrics['r2_x']
    r2_y = metrics['r2_y']
    d_r2 = metrics['d_r2']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_test[:, 0], y_pred[:, 0], s=1, alpha=0.3)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[0].set_xlabel('True X')
    axes[0].set_ylabel('Predicted X')
    axes[0].set_title(f'Position X (R2={r2_x:.3f})')
    axes[0].set_aspect('equal')

    axes[1].scatter(y_test[:, 1], y_pred[:, 1], s=1, alpha=0.3)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[1].set_xlabel('True Y')
    axes[1].set_ylabel('Predicted Y')
    axes[1].set_title(f'Position Y (R2={r2_y:.3f})')
    axes[1].set_aspect('equal')

    axes[2].scatter(d_test, d_pred, s=1, alpha=0.3)
    axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[2].set_xlabel('True d')
    axes[2].set_ylabel('Predicted d')
    axes[2].set_title(f'Curvilinear distance d (R2={d_r2:.3f})')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    _save_fig(fig, '04_scatter_predictions')
    plt.show()


def plot_predictions_with_uncertainty(results):
    """Plot predictions vs true with uncertainty bands (first 500 test points)."""
    y_test = results['y_test']
    y_pred = results['y_pred']
    y_sigma = results['y_sigma']
    eucl_errors = results['eucl_errors']

    segment = slice(0, 500)
    seg_idx = np.arange(500)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 2D trajectory
    colors_pts = np.arange(500)
    axes[0, 0].scatter(y_test[segment, 0], y_test[segment, 1],
                        c=colors_pts, cmap='winter', s=8, alpha=0.6, label='True position')
    sc = axes[0, 0].scatter(y_pred[segment, 0], y_pred[segment, 1],
                             c=colors_pts, cmap='autumn', s=8, alpha=0.6, marker='x', label='Prediction')
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[0, 0].plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.3)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Positions (500 first test points)')
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    cbar = plt.colorbar(sc, ax=axes[0, 0])
    cbar.set_label('Temporal index')

    # 2. X position with uncertainty
    axes[0, 1].plot(seg_idx, y_test[segment, 0], 'b-', label='True X', linewidth=1.5)
    axes[0, 1].plot(seg_idx, y_pred[segment, 0], 'r-', alpha=0.7, label='Prediction', linewidth=1)
    axes[0, 1].fill_between(seg_idx,
                             y_pred[segment, 0] - 2 * y_sigma[segment, 0],
                             y_pred[segment, 0] + 2 * y_sigma[segment, 0],
                             alpha=0.2, color='red', label='Uncertainty (2 sigma)')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Position X')
    axes[0, 1].set_title('Position X with uncertainty')
    axes[0, 1].legend()

    # 3. Y position with uncertainty
    axes[1, 0].plot(seg_idx, y_test[segment, 1], 'b-', label='True Y', linewidth=1.5)
    axes[1, 0].plot(seg_idx, y_pred[segment, 1], 'r-', alpha=0.7, label='Prediction', linewidth=1)
    axes[1, 0].fill_between(seg_idx,
                             y_pred[segment, 1] - 2 * y_sigma[segment, 1],
                             y_pred[segment, 1] + 2 * y_sigma[segment, 1],
                             alpha=0.2, color='red', label='Uncertainty (2 sigma)')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Position Y')
    axes[1, 0].set_title('Position Y with uncertainty')
    axes[1, 0].legend()

    # 4. Calibration: uncertainty vs error
    sigma_mean = (y_sigma[:, 0] + y_sigma[:, 1]) / 2
    axes[1, 1].scatter(sigma_mean, eucl_errors, s=1, alpha=0.3)
    axes[1, 1].set_xlabel('Mean predicted sigma')
    axes[1, 1].set_ylabel('Actual Euclidean error')
    axes[1, 1].set_title('Calibration: uncertainty vs error')
    sigma_range = np.linspace(0, sigma_mean.max(), 100)
    axes[1, 1].plot(sigma_range, 2 * sigma_range, 'r--', label='y = 2*sigma', linewidth=1.5)
    axes[1, 1].legend()

    plt.tight_layout()
    _save_fig(fig, '05_predictions_uncertainty')
    plt.show()

    # Calibration stats
    in_1sigma = np.mean(eucl_errors < sigma_mean)
    in_2sigma = np.mean(eucl_errors < 2 * sigma_mean)
    in_3sigma = np.mean(eucl_errors < 3 * sigma_mean)
    print(f'Uncertainty calibration:')
    print(f'  Error < 1*sigma: {in_1sigma:.1%} (expected ~39% for 2D Gaussian)')
    print(f'  Error < 2*sigma: {in_2sigma:.1%} (expected ~86%)')
    print(f'  Error < 3*sigma: {in_3sigma:.1%} (expected ~99%)')


def plot_feasibility_heatmaps(results):
    """Heatmaps: error, sigma, and distance to skeleton."""
    y_test = results['y_test']
    y_pred = results['y_pred']
    y_sigma = results['y_sigma']
    eucl_errors = results['eucl_errors']
    test_dist_to_skel = results['test_dist_to_skel']
    pct_outside = (test_dist_to_skel > CORRIDOR_HALF_WIDTH).mean()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    sigma_mean = (y_sigma[:, 0] + y_sigma[:, 1]) / 2

    data_list = [
        ('Mean Euclidean error', eucl_errors, 'RdYlGn_r', y_test),
        ('Mean predicted sigma', sigma_mean, 'RdYlGn_r', y_test),
        ('Distance to skeleton (predictions)', test_dist_to_skel, 'Reds', y_pred),
    ]

    for ax_idx, (title, values, cmap, pos) in enumerate(data_list):
        _make_heatmap(axes[ax_idx], pos, values, cmap, title)

    plt.tight_layout()
    _save_fig(fig, '06_feasibility_heatmaps')
    plt.show()

    print(f'Predictions outside maze: {pct_outside:.1%}')
    print(f'Mean distance to skeleton: {test_dist_to_skel.mean():.4f}')


def plot_predictions_distance_to_skeleton(results):
    """Scatter of predictions colored by distance to skeleton + histogram."""
    y_pred = results['y_pred']
    test_dist_to_skel = results['test_dist_to_skel']
    pct_outside = (test_dist_to_skel > CORRIDOR_HALF_WIDTH).mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc = axes[0].scatter(y_pred[:, 0], y_pred[:, 1], c=test_dist_to_skel,
                          cmap='Reds', s=2, alpha=0.5, vmin=0, vmax=0.3)
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[0].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Predictions colored by distance to skeleton')
    axes[0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0], label='Distance to skeleton')

    axes[1].hist(test_dist_to_skel, bins=50, alpha=0.7, edgecolor='white')
    axes[1].axvline(CORRIDOR_HALF_WIDTH, color='red', linestyle='--', linewidth=2,
                     label=f'Corridor threshold ({CORRIDOR_HALF_WIDTH})')
    axes[1].set_xlabel('Distance to skeleton')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Distance distribution (outside: {pct_outside:.1%})')
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, '07_distance_to_skeleton')
    plt.show()


def plot_zone_trajectory(results):
    """Trajectory colored by zone, zone probabilities, curvilinear distance, confusion map."""
    y_pred = results['y_pred']
    y_test = results['y_test']
    zone_pred = results['zone_pred']
    zone_test = results['zone_test']
    probs_ensemble = results['probs_ensemble']
    d_test = results['d_test']
    d_pred = results['d_pred']

    segment = slice(0, 500)
    seg_idx = np.arange(500)
    zone_confusion = zone_pred != zone_test
    zone_confusion_rate = zone_confusion.mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predictions colored by zone
    zone_colors_map = ['blue', 'green', 'red']
    for z in range(N_ZONES):
        zmask = zone_pred == z
        axes[0, 0].scatter(y_pred[zmask, 0], y_pred[zmask, 1],
                            c=zone_colors_map[z], s=2, alpha=0.3, label=ZONE_NAMES[z])
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[0, 0].plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Predictions colored by zone')
    axes[0, 0].legend(markerscale=10)
    axes[0, 0].set_aspect('equal')

    # 2. Zone probabilities over time (500 pts)
    for z in range(N_ZONES):
        axes[0, 1].plot(seg_idx, probs_ensemble[segment, z], label=ZONE_NAMES[z], linewidth=1)
    axes[0, 1].plot(seg_idx, zone_test[segment], 'k--', alpha=0.3, label='True (0/1/2)')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('P(zone) over time (500 pts)')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.05, 1.05)

    # 3. Curvilinear distance predicted vs true
    axes[1, 0].plot(seg_idx, d_test[segment], 'b-', label='True d', linewidth=1.5)
    axes[1, 0].plot(seg_idx, d_pred[segment], 'r-', alpha=0.7, label='Pred d', linewidth=1)
    axes[1, 0].axhline(y=D_LEFT_END, color='blue', linestyle=':', alpha=0.5,
                         label=f'Left threshold ({D_LEFT_END:.3f})')
    axes[1, 0].axhline(y=D_RIGHT_START, color='red', linestyle=':', alpha=0.5,
                         label=f'Right threshold ({D_RIGHT_START:.3f})')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Curvilinear d')
    axes[1, 0].set_title('Curvilinear distance d (500 pts)')
    axes[1, 0].legend()

    # 4. Zone confusion map
    correct = ~zone_confusion
    axes[1, 1].scatter(y_test[correct, 0], y_test[correct, 1], c='green', s=1, alpha=0.2,
                        label=f'Correct ({correct.mean():.1%})')
    if zone_confusion.any():
        axes[1, 1].scatter(y_test[zone_confusion, 0], y_test[zone_confusion, 1],
                            c='red', s=5, alpha=0.8, label=f'Error ({zone_confusion_rate:.1%})')
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[1, 1].plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.3)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Zone confusion (red = error)')
    axes[1, 1].legend(markerscale=5)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    _save_fig(fig, '08_zone_trajectory')
    plt.show()


def plot_zone_heatmaps(results):
    """Heatmaps: predicted zone, Euclidean error, zone confidence."""
    y_test = results['y_test']
    zone_pred = results['zone_pred']
    eucl_errors = results['eucl_errors']
    probs_ensemble = results['probs_ensemble']

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    data_list = [
        ('Predicted zone (argmax)', zone_pred.astype(float), 'RdYlBu'),
        ('Euclidean error', eucl_errors, 'RdYlGn_r'),
        ('Zone confidence (max prob)', probs_ensemble.max(axis=1), 'RdYlGn'),
    ]

    for ax_idx, (title, values, cmap) in enumerate(data_list):
        _make_heatmap(axes[ax_idx], y_test, values, cmap, title)

    plt.tight_layout()
    _save_fig(fig, '09_zone_heatmaps')
    plt.show()


def plot_confusion_matrix(results):
    """Confusion matrix (raw counts + normalized) for 3 zones."""
    zone_test = results['zone_test']
    zone_pred = results['zone_pred']
    cls_accuracy = (zone_pred == zone_test).mean()

    cm = confusion_matrix(zone_test, zone_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    im0 = axes[0].imshow(cm, cmap='Blues')
    for i in range(N_ZONES):
        for j in range(N_ZONES):
            axes[0].text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=14,
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
    axes[0].set_xticks(range(N_ZONES))
    axes[0].set_xticklabels(ZONE_NAMES)
    axes[0].set_yticks(range(N_ZONES))
    axes[0].set_yticklabels(ZONE_NAMES)
    axes[0].set_xlabel('Predicted zone')
    axes[0].set_ylabel('True zone')
    axes[0].set_title('Confusion matrix (counts)')
    plt.colorbar(im0, ax=axes[0])

    # Normalized
    im1 = axes[1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(N_ZONES):
        for j in range(N_ZONES):
            axes[1].text(j, i, f'{cm_norm[i, j]:.1%}', ha='center', va='center', fontsize=14,
                         color='white' if cm_norm[i, j] > 0.5 else 'black')
    axes[1].set_xticks(range(N_ZONES))
    axes[1].set_xticklabels(ZONE_NAMES)
    axes[1].set_yticks(range(N_ZONES))
    axes[1].set_yticklabels(ZONE_NAMES)
    axes[1].set_xlabel('Predicted zone')
    axes[1].set_ylabel('True zone')
    axes[1].set_title('Confusion matrix (normalized)')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    _save_fig(fig, '10_confusion_matrix')
    plt.show()

    print(f'Global accuracy: {cls_accuracy:.1%}')
    for z in range(N_ZONES):
        print(f'  {ZONE_NAMES[z]:8s}: recall={cm_norm[z, z]:.1%}')


def plot_interfold_variance(results):
    """Analyze inter-fold variance (epistemic vs aleatoric uncertainty)."""
    all_fold_mu = results['all_fold_mu']
    all_fold_sigma = results['all_fold_sigma']
    y_test = results['y_test']
    eucl_errors = results['eucl_errors']
    var_mu = results['var_mu']

    fold_std_mu = all_fold_mu.std(axis=0)
    fold_std_eucl = np.sqrt(fold_std_mu[:, 0]**2 + fold_std_mu[:, 1]**2)

    aleatoric = (all_fold_sigma ** 2).mean(axis=0)
    aleatoric_mean = np.sqrt(aleatoric).mean(axis=1)
    epistemic = np.sqrt(var_mu).mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Inter-fold variance by position
    sc = axes[0, 0].scatter(y_test[:, 0], y_test[:, 1], c=fold_std_eucl, cmap='Reds', s=2, alpha=0.5)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Inter-fold variance (epistemic)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 0], label='Std of mu across folds')

    # 2. Aleatoric vs Epistemic
    axes[0, 1].scatter(aleatoric_mean, epistemic, s=1, alpha=0.3)
    max_val = max(aleatoric_mean.max(), epistemic.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='y=x')
    axes[0, 1].set_xlabel('Aleatoric uncertainty (mean sigma)')
    axes[0, 1].set_ylabel('Epistemic uncertainty (std mu across folds)')
    axes[0, 1].set_title('Aleatoric vs Epistemic')
    axes[0, 1].legend()

    # 3. Distribution of both uncertainty types
    axes[1, 0].hist(aleatoric_mean, bins=50, alpha=0.7, label='Aleatoric', color='steelblue')
    axes[1, 0].hist(epistemic, bins=50, alpha=0.7, label='Epistemic', color='coral')
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Uncertainty distributions')
    axes[1, 0].legend()

    # 4. Error vs total uncertainty
    total_uncertainty = np.sqrt(aleatoric_mean**2 + epistemic**2)
    axes[1, 1].scatter(total_uncertainty, eucl_errors, s=1, alpha=0.3)
    u_range = np.linspace(0, total_uncertainty.max(), 100)
    axes[1, 1].plot(u_range, 2 * u_range, 'r--', label='y = 2*sigma_total')
    axes[1, 1].set_xlabel('Total uncertainty (sqrt(aleatoric^2 + epistemic^2))')
    axes[1, 1].set_ylabel('Euclidean error')
    axes[1, 1].set_title('Total calibration')
    axes[1, 1].legend()

    plt.tight_layout()
    _save_fig(fig, '11_interfold_variance')
    plt.show()

    corr_alea, _ = spearmanr(aleatoric_mean, eucl_errors)
    corr_epis, _ = spearmanr(epistemic, eucl_errors)
    corr_total, _ = spearmanr(total_uncertainty, eucl_errors)
    print(f'Spearman correlation (uncertainty vs error):')
    print(f'  Aleatoric : {corr_alea:.3f}')
    print(f'  Epistemic : {corr_epis:.3f}')
    print(f'  Total     : {corr_total:.3f}')


def plot_fold_agreement(results):
    """Analyze zone agreement across 5 folds."""
    all_fold_probs = results['all_fold_probs']
    y_test = results['y_test']

    fold_zone_preds = np.stack([fp.argmax(axis=1) for fp in all_fold_probs])
    zone_agreement = np.zeros(len(y_test))
    for i in range(len(y_test)):
        most_common = np.bincount(fold_zone_preds[:, i], minlength=3).max()
        zone_agreement[i] = most_common / N_FOLDS

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Spatial agreement
    sc = axes[0].scatter(y_test[:, 0], y_test[:, 1], c=zone_agreement,
                          cmap='RdYlGn', s=2, alpha=0.5, vmin=0.4, vmax=1.0)
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        axes[0].plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Inter-fold zone agreement (5 folds)')
    axes[0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0], label='% agreement (5 folds)')

    # 2. Agreement histogram
    unique_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    counts = [(zone_agreement == v).sum() for v in unique_vals]
    labels = ['1/5', '2/5', '3/5', '4/5', '5/5']
    bars = axes[1].bar(labels, counts, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    for bar, c in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f'{c}', ha='center')
    axes[1].set_xlabel('Agreement (out of 5 folds)')
    axes[1].set_ylabel('Number of points')
    axes[1].set_title('Distribution of zone agreement across folds')

    plt.tight_layout()
    _save_fig(fig, '12_fold_agreement')
    plt.show()

    print(f'Perfect agreement (5/5): {(zone_agreement == 1.0).mean():.1%}')
    print(f'Majority agreement (>=3/5): {(zone_agreement >= 0.6).mean():.1%}')


def plot_sigma_distribution(results):
    """Distribution of predicted sigma for X and Y."""
    y_sigma = results['y_sigma']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_sigma[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(y_sigma[:, 0].mean(), color='red', linestyle='--',
                     label=f'Mean={y_sigma[:, 0].mean():.4f}')
    axes[0].set_xlabel('Sigma X')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Uncertainty distribution for X')
    axes[0].legend()

    axes[1].hist(y_sigma[:, 1], bins=50, alpha=0.7, color='coral', edgecolor='white')
    axes[1].axvline(y_sigma[:, 1].mean(), color='red', linestyle='--',
                     label=f'Mean={y_sigma[:, 1].mean():.4f}')
    axes[1].set_xlabel('Sigma Y')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Uncertainty distribution for Y')
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, '13_sigma_distribution')
    plt.show()


def plot_uncertainty_heatmaps(results):
    """Heatmaps: error and sigma by spatial position."""
    y_test = results['y_test']
    y_sigma = results['y_sigma']
    eucl_errors = results['eucl_errors']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sigma_mean = (y_sigma[:, 0] + y_sigma[:, 1]) / 2
    _make_heatmap(axes[0], y_test, eucl_errors, 'RdYlGn_r', 'Mean Euclidean error')
    _make_heatmap(axes[1], y_test, sigma_mean, 'RdYlGn_r', 'Mean predicted sigma')

    plt.tight_layout()
    _save_fig(fig, '14_uncertainty_heatmaps')
    plt.show()

    corr, pval = spearmanr(sigma_mean, eucl_errors)
    print(f'Spearman correlation (sigma vs error): {corr:.3f} (p={pval:.2e})')
    if corr > 0.3:
        print(f'  -> Good calibration: model knows when it makes errors')
    else:
        print(f'  -> Weak calibration: model does not know well when it errs')


def run_all_visualizations(results, metrics, all_train_losses=None,
                           all_val_losses=None, all_train_losses_detail=None,
                           all_val_losses_detail=None):
    """Run all visualization functions."""
    if all_train_losses is not None:
        plot_training_curves(all_train_losses, all_val_losses)
        plot_loss_decomposition(all_train_losses_detail, all_val_losses_detail)

    plot_scatter_predictions(results, metrics)
    plot_predictions_with_uncertainty(results)
    plot_feasibility_heatmaps(results)
    plot_predictions_distance_to_skeleton(results)
    plot_zone_trajectory(results)
    plot_zone_heatmaps(results)
    plot_confusion_matrix(results)
    plot_interfold_variance(results)
    plot_fold_agreement(results)
    plot_sigma_distribution(results)
    plot_uncertainty_heatmaps(results)
