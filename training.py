import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from config import (
    SEED, DEVICE, EMBED_DIM, NHEAD, NUM_LAYERS, DROPOUT, SPIKE_DROPOUT,
    NOISE_STD, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE, N_FOLDS,
    LAMBDA_D, LAMBDA_FEAS, OUTPUT_DIR,
)
from geometry import (
    SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH, N_ZONES, ZONE_NAMES,
    compute_distance_to_skeleton,
)
from data import SpikeSequenceDataset, collate_fn
from model import SpikeTransformerHierarchical
from losses import FeasibilityLoss


def train_epoch(model, loader, optimizer, scheduler, criterion_ce,
                criterion_nll, criterion_d, feas_loss, device):
    """Train for one epoch. Returns (loss_dict, accuracy)."""
    model.train()
    totals = {'loss': 0, 'cls': 0, 'pos': 0, 'd': 0, 'feas': 0, 'correct': 0, 'n': 0, 'batches': 0}

    for batch in loader:
        wf = batch['waveforms'].to(device)
        sid = batch['shank_ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        d_targets = batch['d_targets'].to(device)
        zone_targets = batch['zone_targets'].to(device)

        optimizer.zero_grad()
        cls_logits, mus, sigmas, d_pred = model(wf, sid, mask)

        loss_cls = criterion_ce(cls_logits, zone_targets)

        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = (zone_targets == z)
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(
                    mus[z][zmask], targets[zmask], sigmas[z][zmask] ** 2
                )

        loss_d = criterion_d(d_pred.squeeze(-1), d_targets)

        probs = torch.softmax(cls_logits, dim=1).unsqueeze(-1)
        mu_stack = torch.stack(mus, dim=1)
        mu_combined = (probs * mu_stack).sum(dim=1)
        loss_feas = feas_loss(mu_combined)

        loss = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        totals['loss'] += loss.item()
        totals['cls'] += loss_cls.item()
        totals['pos'] += loss_pos.item()
        totals['d'] += loss_d.item()
        totals['feas'] += loss_feas.item()
        with torch.no_grad():
            totals['correct'] += (cls_logits.argmax(dim=1) == zone_targets).sum().item()
            totals['n'] += len(zone_targets)
        totals['batches'] += 1

    nb = totals['batches']
    return {k: totals[k] / nb for k in ['loss', 'cls', 'pos', 'd', 'feas']}, totals['correct'] / totals['n']


@torch.no_grad()
def eval_epoch(model, loader, criterion_ce, criterion_nll, criterion_d,
               feas_loss, device):
    """Evaluate for one epoch. Returns (loss_dict, accuracy, prediction_arrays)."""
    model.eval()
    totals = {'loss': 0, 'cls': 0, 'pos': 0, 'd': 0, 'feas': 0, 'correct': 0, 'n': 0, 'batches': 0}
    all_mu, all_sigma, all_probs, all_d = [], [], [], []
    all_targets, all_d_targets, all_zone_targets = [], [], []

    for batch in loader:
        wf = batch['waveforms'].to(device)
        sid = batch['shank_ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        d_targets = batch['d_targets'].to(device)
        zone_targets = batch['zone_targets'].to(device)

        mu, sigma, probs, d_pred = model.predict(wf, sid, mask)
        cls_logits, mus, sigmas_z, _ = model(wf, sid, mask)

        loss_cls = criterion_ce(cls_logits, zone_targets)
        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = (zone_targets == z)
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(
                    mus[z][zmask], targets[zmask], sigmas_z[z][zmask] ** 2
                )
        loss_d = criterion_d(d_pred.squeeze(-1), d_targets)
        loss_feas = feas_loss(mu)
        loss = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas

        totals['loss'] += loss.item()
        totals['cls'] += loss_cls.item()
        totals['pos'] += loss_pos.item()
        totals['d'] += loss_d.item()
        totals['feas'] += loss_feas.item()
        totals['correct'] += (cls_logits.argmax(dim=1) == zone_targets).sum().item()
        totals['n'] += len(zone_targets)
        totals['batches'] += 1

        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_d.append(d_pred.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_d_targets.append(d_targets.cpu().numpy())
        all_zone_targets.append(zone_targets.cpu().numpy())

    nb = totals['batches']
    losses = {k: totals[k] / nb for k in ['loss', 'cls', 'pos', 'd', 'feas']}
    acc = totals['correct'] / totals['n']
    arrays = tuple(np.concatenate(a) for a in [
        all_mu, all_sigma, all_probs, all_d, all_targets, all_d_targets, all_zone_targets
    ])
    return losses, acc, arrays


def train_all_folds(df_train_full, d_train_full, zone_train_full,
                    nGroups, nChannelsPerGroup, max_channels):
    """Run 5-fold cross-validation training.

    Returns:
        fold_results: list of per-fold metric dicts
        all_train_losses: dict[fold] -> list of total train loss per epoch
        all_val_losses: dict[fold] -> list of total val loss per epoch
        all_train_losses_detail: dict[fold] -> list of loss dicts per epoch
        all_val_losses_detail: dict[fold] -> list of loss dicts per epoch
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=41)

    fold_results = []
    all_train_losses = {}
    all_val_losses = {}
    all_train_losses_detail = {}
    all_val_losses_detail = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_full)):
        print(f'\n{"=" * 60}')
        print(f'FOLD {fold + 1}/{N_FOLDS}')
        print(f'{"=" * 60}')

        df_ft = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_fv = df_train_full.iloc[val_idx].reset_index(drop=True)

        ds_t = SpikeSequenceDataset(
            df_ft, nGroups, nChannelsPerGroup,
            d_train_full[train_idx], zone_train_full[train_idx],
            max_channels=max_channels,
        )
        ds_v = SpikeSequenceDataset(
            df_fv, nGroups, nChannelsPerGroup,
            d_train_full[val_idx], zone_train_full[val_idx],
            max_channels=max_channels,
        )
        dl_t = DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
        dl_v = DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

        print(f'  Train: {len(ds_t)}, Val: {len(ds_v)}')

        model = SpikeTransformerHierarchical(
            nGroups, nChannelsPerGroup, n_zones=N_ZONES,
            embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
            dropout=DROPOUT, spike_dropout=SPIKE_DROPOUT, noise_std=NOISE_STD,
            max_channels=max_channels,
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(dl_t),
        )
        criterion_ce = nn.CrossEntropyLoss()
        criterion_nll = nn.GaussianNLLLoss()
        criterion_d = nn.MSELoss()
        feas_loss_fn = FeasibilityLoss(SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH).to(DEVICE)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        train_detail, val_detail = [], []
        model_path = f'{OUTPUT_DIR}/best_transformer_02i_fold{fold + 1}.pt'

        for epoch in range(EPOCHS):
            t_losses, t_acc = train_epoch(
                model, dl_t, optimizer, scheduler,
                criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
            )
            v_losses, v_acc, _ = eval_epoch(
                model, dl_v, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
            )

            train_losses.append(t_losses['loss'])
            val_losses.append(v_losses['loss'])
            train_detail.append(t_losses)
            val_detail.append(v_losses)

            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(
                    f'  Epoch {epoch + 1:02d}/{EPOCHS} | '
                    f'Train: {t_losses["loss"]:.4f} '
                    f'(cls={t_losses["cls"]:.4f} pos={t_losses["pos"]:.4f} '
                    f'd={t_losses["d"]:.5f} feas={t_losses["feas"]:.6f} acc={t_acc:.1%}) | '
                    f'Val: {v_losses["loss"]:.4f} (acc={v_acc:.1%})'
                )

            if v_losses['loss'] < best_val_loss:
                best_val_loss = v_losses['loss']
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break

        all_train_losses[fold] = train_losses
        all_val_losses[fold] = val_losses
        all_train_losses_detail[fold] = train_detail
        all_val_losses_detail[fold] = val_detail

        # Evaluate on this fold's validation set
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        _, val_acc, (val_mu, val_sigma, val_probs, val_d_pred,
                     val_targets, val_d_targets, val_zone_targets) = eval_epoch(
            model, dl_v, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
        )
        val_eucl = np.sqrt(((val_targets - val_mu) ** 2).sum(axis=1))
        val_d_mae = np.abs(val_d_targets - val_d_pred.squeeze()).mean()

        val_dist_to_skel = np.array([
            compute_distance_to_skeleton(val_mu[i, 0], val_mu[i, 1])
            for i in range(len(val_mu))
        ])
        pct_outside = (val_dist_to_skel > CORRIDOR_HALF_WIDTH).mean()

        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'val_eucl_mean': val_eucl.mean(),
            'val_r2_x': r2_score(val_targets[:, 0], val_mu[:, 0]),
            'val_r2_y': r2_score(val_targets[:, 1], val_mu[:, 1]),
            'val_d_mae': val_d_mae,
            'val_cls_acc': val_acc,
            'val_pct_outside': pct_outside,
            'epochs': len(train_losses),
        })
        print(
            f'  => Eucl={val_eucl.mean():.4f} | '
            f'R2: X={fold_results[-1]["val_r2_x"]:.4f} Y={fold_results[-1]["val_r2_y"]:.4f} | '
            f'd_MAE={val_d_mae:.4f} | cls={val_acc:.1%} | outside={pct_outside:.1%}'
        )

    # Summary
    print(f'\n{"=" * 60}')
    print(f'CROSS-VALIDATION SUMMARY ({N_FOLDS} folds)')
    print(f'{"=" * 60}')
    for r in fold_results:
        print(
            f'  Fold {r["fold"]}: Eucl={r["val_eucl_mean"]:.4f} | '
            f'R2_X={r["val_r2_x"]:.4f} R2_Y={r["val_r2_y"]:.4f} | '
            f'd_MAE={r["val_d_mae"]:.4f} | cls={r["val_cls_acc"]:.1%} | '
            f'outside={r["val_pct_outside"]:.1%} | epochs={r["epochs"]}'
        )
    print(
        f'\n  Mean: Eucl={np.mean([r["val_eucl_mean"] for r in fold_results]):.4f} '
        f'(+/- {np.std([r["val_eucl_mean"] for r in fold_results]):.4f})'
    )
    print(
        f'        R2_X={np.mean([r["val_r2_x"] for r in fold_results]):.4f} | '
        f'R2_Y={np.mean([r["val_r2_y"] for r in fold_results]):.4f}'
    )
    print(
        f'        cls={np.mean([r["val_cls_acc"] for r in fold_results]):.1%} | '
        f'outside={np.mean([r["val_pct_outside"] for r in fold_results]):.1%}'
    )

    return fold_results, all_train_losses, all_val_losses, all_train_losses_detail, all_val_losses_detail


def ensemble_evaluate(test_loader, nGroups, nChannelsPerGroup, max_channels):
    """Load all fold checkpoints and compute ensemble predictions on test set.

    Returns dict with all predictions, targets, metrics, and per-fold arrays.
    """
    criterion_ce = nn.CrossEntropyLoss()
    criterion_nll = nn.GaussianNLLLoss()
    criterion_d = nn.MSELoss()
    feas_loss_fn = FeasibilityLoss(SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH).to(DEVICE)

    all_fold_mu, all_fold_sigma, all_fold_probs, all_fold_d = [], [], [], []

    for fold in range(N_FOLDS):
        model = SpikeTransformerHierarchical(
            nGroups, nChannelsPerGroup, n_zones=N_ZONES,
            embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
            dropout=DROPOUT, spike_dropout=SPIKE_DROPOUT, noise_std=NOISE_STD,
            max_channels=max_channels,
        ).to(DEVICE)
        model.load_state_dict(torch.load(
            f'{OUTPUT_DIR}/best_transformer_02i_fold{fold + 1}.pt',
            map_location=DEVICE, weights_only=True,
        ))

        _, fold_acc, (fold_mu, fold_sigma, fold_probs, fold_d,
                      y_test, d_test_targets, zone_test_targets) = eval_epoch(
            model, test_loader, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
        )
        all_fold_mu.append(fold_mu)
        all_fold_sigma.append(fold_sigma)
        all_fold_probs.append(fold_probs)
        all_fold_d.append(fold_d)
        fold_eucl = np.sqrt(((y_test - fold_mu) ** 2).sum(axis=1))
        print(f'Fold {fold + 1}: Eucl={fold_eucl.mean():.4f}, cls_acc={fold_acc:.1%}')

    # Stack fold predictions
    all_fold_mu = np.stack(all_fold_mu)
    all_fold_sigma = np.stack(all_fold_sigma)
    all_fold_probs = np.stack(all_fold_probs)
    all_fold_d = np.stack(all_fold_d)

    # Ensemble aggregation
    y_pred = all_fold_mu.mean(axis=0)
    d_pred_ensemble = all_fold_d.mean(axis=0).squeeze()
    probs_ensemble = all_fold_probs.mean(axis=0)
    zone_pred = probs_ensemble.argmax(axis=1)

    # Total variance (law of total variance)
    mean_var = (all_fold_sigma ** 2).mean(axis=0)
    var_mu = all_fold_mu.var(axis=0)
    y_sigma = np.sqrt(mean_var + var_mu)

    # Euclidean errors
    eucl_errors = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1))

    # Distance to skeleton
    test_dist_to_skel = np.array([
        compute_distance_to_skeleton(y_pred[i, 0], y_pred[i, 1])
        for i in range(len(y_pred))
    ])

    return {
        'y_pred': y_pred,
        'y_sigma': y_sigma,
        'y_test': y_test,
        'd_pred': d_pred_ensemble,
        'd_test': d_test_targets,
        'zone_pred': zone_pred,
        'zone_test': zone_test_targets,
        'probs_ensemble': probs_ensemble,
        'eucl_errors': eucl_errors,
        'test_dist_to_skel': test_dist_to_skel,
        'var_mu': var_mu,
        'all_fold_mu': all_fold_mu,
        'all_fold_sigma': all_fold_sigma,
        'all_fold_probs': all_fold_probs,
    }


def print_metrics(results):
    """Print all evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_test = results['y_test']
    y_pred = results['y_pred']
    y_sigma = results['y_sigma']
    d_test = results['d_test']
    d_pred = results['d_pred']
    zone_pred = results['zone_pred']
    zone_test = results['zone_test']
    eucl_errors = results['eucl_errors']
    test_dist_to_skel = results['test_dist_to_skel']

    mse_x = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_y = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
    d_mae = np.abs(d_test - d_pred).mean()
    d_r2 = r2_score(d_test, d_pred)
    cls_accuracy = (zone_pred == zone_test).mean()
    pct_outside = (test_dist_to_skel > CORRIDOR_HALF_WIDTH).mean()

    print(f'\n{"=" * 60}')
    print(f'02i Combined — Ensemble ({N_FOLDS} folds)')
    print(f'{"=" * 60}')
    print(f'  MSE  : X={mse_x:.5f}, Y={mse_y:.5f}')
    print(f'  MAE  : X={mae_x:.4f}, Y={mae_y:.4f}')
    print(f'  R2   : X={r2_x:.4f}, Y={r2_y:.4f}')
    print(f'  Eucl : mean={eucl_errors.mean():.4f}, '
          f'median={np.median(eucl_errors):.4f}, '
          f'p90={np.percentile(eucl_errors, 90):.4f}')
    print(f'\n  Curvilinear d: MAE={d_mae:.4f}, R2={d_r2:.4f}')
    print(f'  Zone classification: accuracy={cls_accuracy:.1%}')
    print(f'  Outside maze: {pct_outside:.1%}')

    print(f'\n  Error per zone:')
    for z in range(N_ZONES):
        zmask = zone_test == z
        if zmask.any():
            z_acc = (zone_pred[zmask] == z).mean()
            print(f'    {ZONE_NAMES[z]:8s} : Eucl={eucl_errors[zmask].mean():.4f} | '
                  f'cls_acc={z_acc:.1%} ({zmask.sum()} pts)')

    print(f'\n  Mean sigma: X={y_sigma[:, 0].mean():.4f}, Y={y_sigma[:, 1].mean():.4f}')

    return {
        'mse_x': mse_x, 'mse_y': mse_y,
        'mae_x': mae_x, 'mae_y': mae_y,
        'r2_x': r2_x, 'r2_y': r2_y,
        'd_mae': d_mae, 'd_r2': d_r2,
        'cls_accuracy': cls_accuracy,
        'pct_outside': pct_outside,
    }
