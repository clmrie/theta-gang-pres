"""
Spike Transformer Hierarchical — U-maze position decoder

Trains a hierarchical Transformer model to predict mouse position in a U-maze
from neural spike recordings. Combines:
  - Spike dropout (15%) + Gaussian noise (std=0.5) augmentation
  - Feasibility loss (penalizes predictions outside corridor)
  - 3-zone hierarchical classification + conditional regression
  - 5-fold cross-validation with ensemble evaluation

Usage:
    python main.py                  # Full pipeline: train + evaluate + visualize
    python main.py --skip-training  # Load checkpoints and evaluate + visualize only
"""

import argparse
import os
import warnings

import numpy as np
from torch.utils.data import DataLoader

from config import SEED, DEVICE, BATCH_SIZE, N_FOLDS, OUTPUT_DIR, FIGURES_DIR
from geometry import (
    N_ZONES, ZONE_NAMES, CORRIDOR_HALF_WIDTH,
    compute_curvilinear_distance, compute_distance_to_skeleton, compute_all_geometry,
)
from data import load_data, SpikeSequenceDataset, collate_fn
from training import train_all_folds, ensemble_evaluate, print_metrics
from visualization import plot_umaze_geometry, run_all_visualizations

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Spike Transformer Hierarchical')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, load existing checkpoints')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Seed: {SEED}')

    # ── 1. Load data ──────────────────────────────────────────────
    df_moving, nGroups, nChannelsPerGroup = load_data()
    max_channels = max(nChannelsPerGroup)

    # ── 2. Compute geometry ───────────────────────────────────────
    positions = np.array([[x[0], x[1]] for x in df_moving['pos']], dtype=np.float32)
    curvilinear_d, zone_labels = compute_all_geometry(positions)

    print(f'\nCurvilinear d: min={curvilinear_d.min():.4f}, '
          f'max={curvilinear_d.max():.4f}, mean={curvilinear_d.mean():.4f}')
    print(f'\nZone distribution:')
    for z in range(N_ZONES):
        count = (zone_labels == z).sum()
        print(f'  {ZONE_NAMES[z]:8s} (class {z}): {count} ({count / len(zone_labels):.1%})')

    dist_to_skel = np.array([compute_distance_to_skeleton(x, y) for x, y in positions])
    print(f'\nDistance to skeleton: mean={dist_to_skel.mean():.4f}, max={dist_to_skel.max():.4f}')
    print(f'  % inside corridor (dist < {CORRIDOR_HALF_WIDTH}): '
          f'{(dist_to_skel < CORRIDOR_HALF_WIDTH).mean():.1%}')

    # ── 3. Plot U-maze geometry ───────────────────────────────────
    plot_umaze_geometry(positions, curvilinear_d, zone_labels)

    # ── 4. Train/test split ───────────────────────────────────────
    split_idx = int(len(df_moving) * 0.9)
    df_train_full = df_moving.iloc[:split_idx].reset_index(drop=True)
    df_test = df_moving.iloc[split_idx:].reset_index(drop=True)
    d_train_full = curvilinear_d[:split_idx]
    d_test = curvilinear_d[split_idx:]
    zone_train_full = zone_labels[:split_idx]
    zone_test = zone_labels[split_idx:]

    print(f'\nTrain: {len(df_train_full)} | Test: {len(df_test)}')
    for z in range(N_ZONES):
        print(f'  {ZONE_NAMES[z]:8s} — train: {(zone_train_full == z).sum()}, '
              f'test: {(zone_test == z).sum()}')

    test_dataset = SpikeSequenceDataset(
        df_test, nGroups, nChannelsPerGroup, d_test, zone_test, max_channels=max_channels,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f'Test: {len(test_dataset)} examples, {len(test_loader)} batches')

    # ── 5. Training ───────────────────────────────────────────────
    all_train_losses = None
    all_val_losses = None
    all_train_losses_detail = None
    all_val_losses_detail = None

    if not args.skip_training:
        (fold_results, all_train_losses, all_val_losses,
         all_train_losses_detail, all_val_losses_detail) = train_all_folds(
            df_train_full, d_train_full, zone_train_full,
            nGroups, nChannelsPerGroup, max_channels,
        )

    # ── 6. Ensemble evaluation ────────────────────────────────────
    print('\n' + '=' * 60)
    print('ENSEMBLE EVALUATION ON TEST SET')
    print('=' * 60)
    results = ensemble_evaluate(test_loader, nGroups, nChannelsPerGroup, max_channels)
    metrics = print_metrics(results)

    # ── 7. All visualizations ─────────────────────────────────────
    run_all_visualizations(
        results, metrics,
        all_train_losses, all_val_losses,
        all_train_losses_detail, all_val_losses_detail,
    )

    # ── 8. Save predictions ───────────────────────────────────────
    np.save(f'{OUTPUT_DIR}/preds_transformer_02i.npy', results['y_pred'])
    np.save(f'{OUTPUT_DIR}/sigma_transformer_02i.npy', results['y_sigma'])
    np.save(f'{OUTPUT_DIR}/d_pred_transformer_02i.npy', results['d_pred'])
    np.save(f'{OUTPUT_DIR}/y_test_transformer_02i.npy', results['y_test'])
    np.save(f'{OUTPUT_DIR}/d_test_transformer_02i.npy', results['d_test'])
    np.save(f'{OUTPUT_DIR}/zone_pred_transformer_02i.npy', results['zone_pred'])
    np.save(f'{OUTPUT_DIR}/zone_test_transformer_02i.npy', results['zone_test'])
    np.save(f'{OUTPUT_DIR}/probs_transformer_02i.npy', results['probs_ensemble'])

    print(f'\nPredictions saved to {OUTPUT_DIR}/:')
    print(f'  preds_transformer_02i.npy : {results["y_pred"].shape}')
    print(f'  sigma_transformer_02i.npy : {results["y_sigma"].shape}')
    print(f'  d_pred_transformer_02i.npy: {results["d_pred"].shape}')
    print(f'  zone_pred_transformer_02i.npy: {results["zone_pred"].shape}')
    print(f'  probs_transformer_02i.npy : {results["probs_ensemble"].shape}')
    print(f'\nFigures saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
