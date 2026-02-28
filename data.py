import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_DIR, PARQUET_NAME, JSON_NAME, MAX_SEQ_LEN


def load_data(data_dir=DATA_DIR):
    """Load parquet and JSON config, filter by speed mask.

    Returns:
        df_moving: DataFrame with only moving examples
        nGroups: number of electrode groups
        nChannelsPerGroup: list of channel counts per group
    """
    parquet_path = os.path.join(data_dir, PARQUET_NAME)
    json_path = os.path.join(data_dir, JSON_NAME)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Data not found at {data_dir}/\n"
            f"Expected: {PARQUET_NAME}"
        )

    print(f"Loading data from {data_dir}/")
    df = pd.read_parquet(parquet_path)
    with open(json_path, "r") as f:
        params = json.load(f)

    print(f"Shape: {df.shape}")

    nGroups = params['nGroups']
    nChannelsPerGroup = [params[f'group{g}']['nChannels'] for g in range(nGroups)]
    print(f"nGroups={nGroups}, nChannelsPerGroup={nChannelsPerGroup}")

    # Filter by speed mask (keep only moving examples)
    speed_masks = np.array([x[0] for x in df['speedMask']])
    df_moving = df[speed_masks].reset_index(drop=True)
    print(f'Moving examples: {len(df_moving)}')

    return df_moving, nGroups, nChannelsPerGroup


def reconstruct_sequence(row, nGroups, nChannelsPerGroup, max_seq_len=MAX_SEQ_LEN):
    """Reconstruct chronological spike sequence from a dataframe row.

    Returns:
        seq_waveforms: list of (waveform_array, group_id) tuples
        seq_shank_ids: list of group IDs for each spike
    """
    groups = row['groups']
    length = min(len(groups), max_seq_len)
    waveforms = {}
    for g in range(nGroups):
        nCh = nChannelsPerGroup[g]
        raw = row[f'group{g}']
        waveforms[g] = raw.reshape(-1, nCh, 32)

    seq_waveforms = []
    seq_shank_ids = []
    for t in range(length):
        g = int(groups[t])
        idx = int(row[f'indices{g}'][t])
        if idx > 0 and idx <= waveforms[g].shape[0]:
            seq_waveforms.append((waveforms[g][idx - 1], g))
            seq_shank_ids.append(g)
    return seq_waveforms, seq_shank_ids


class SpikeSequenceDataset(Dataset):
    """PyTorch dataset for spike sequences with zone labels and curvilinear distance."""

    def __init__(self, dataframe, nGroups, nChannelsPerGroup, curvilinear_d,
                 zone_labels, max_seq_len=MAX_SEQ_LEN, max_channels=None):
        self.df = dataframe
        self.nGroups = nGroups
        self.nChannelsPerGroup = nChannelsPerGroup
        self.max_seq_len = max_seq_len
        self.max_channels = max_channels or max(nChannelsPerGroup)
        self.targets = np.array([[x[0], x[1]] for x in dataframe['pos']], dtype=np.float32)
        self.curvilinear_d = curvilinear_d.astype(np.float32)
        self.zone_labels = zone_labels.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq, shank_ids = reconstruct_sequence(
            row, self.nGroups, self.nChannelsPerGroup, self.max_seq_len
        )
        seq_len = len(seq)
        if seq_len == 0:
            seq_len = 1
            waveforms = np.zeros((1, self.max_channels, 32), dtype=np.float32)
            shank_ids_arr = np.array([0], dtype=np.int64)
        else:
            waveforms = np.zeros((seq_len, self.max_channels, 32), dtype=np.float32)
            shank_ids_arr = np.array(shank_ids, dtype=np.int64)
            for t, (wf, g) in enumerate(seq):
                nCh = wf.shape[0]
                waveforms[t, :nCh, :] = wf
        return {
            'waveforms': torch.from_numpy(waveforms),
            'shank_ids': torch.from_numpy(shank_ids_arr),
            'seq_len': seq_len,
            'target': torch.from_numpy(self.targets[idx]),
            'd': torch.tensor(self.curvilinear_d[idx], dtype=torch.float32),
            'zone': torch.tensor(self.zone_labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length spike sequences."""
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    max_channels = batch[0]['waveforms'].shape[1]

    waveforms = torch.zeros(batch_size, max_len, max_channels, 32)
    shank_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    targets = torch.stack([item['target'] for item in batch])
    d_targets = torch.stack([item['d'] for item in batch])
    zone_targets = torch.stack([item['zone'] for item in batch])

    for i, item in enumerate(batch):
        sl = item['seq_len']
        waveforms[i, :sl] = item['waveforms']
        shank_ids[i, :sl] = item['shank_ids']
        mask[i, :sl] = False

    return {
        'waveforms': waveforms,
        'shank_ids': shank_ids,
        'mask': mask,
        'targets': targets,
        'd_targets': d_targets,
        'zone_targets': zone_targets,
    }
