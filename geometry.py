import numpy as np

# U-maze skeleton: 3 segments forming a U shape
SKELETON_SEGMENTS = np.array([
    [0.15, 0.0, 0.15, 0.85],   # Segment 1: left arm (bottom -> top)
    [0.15, 0.85, 0.85, 0.85],  # Segment 2: top corridor (left -> right)
    [0.85, 0.85, 0.85, 0.0],   # Segment 3: right arm (top -> bottom)
])

CORRIDOR_HALF_WIDTH = 0.15

# Derived constants
SEGMENT_LENGTHS = np.array([
    np.sqrt((s[2] - s[0])**2 + (s[3] - s[1])**2) for s in SKELETON_SEGMENTS
])
TOTAL_LENGTH = SEGMENT_LENGTHS.sum()  # ~2.40
CUMULATIVE_LENGTHS = np.concatenate([[0], np.cumsum(SEGMENT_LENGTHS)])

# Zone thresholds
D_LEFT_END = CUMULATIVE_LENGTHS[1] / TOTAL_LENGTH    # ~0.354
D_RIGHT_START = CUMULATIVE_LENGTHS[2] / TOTAL_LENGTH  # ~0.646

N_ZONES = 3
ZONE_NAMES = ['Left', 'Top', 'Right']


def project_point_on_segment(px, py, x1, y1, x2, y2):
    """Project point (px, py) onto segment [(x1,y1), (x2,y2)].

    Returns (t, dist, proj_x, proj_y) where t is the parameter along the segment.
    """
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx**2 + dy**2
    if seg_len_sq < 1e-12:
        return 0.0, np.sqrt((px - x1)**2 + (py - y1)**2), x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return t, dist, proj_x, proj_y


def compute_curvilinear_distance(x, y):
    """Normalized curvilinear distance d in [0, 1] along the U-maze."""
    best_dist = np.inf
    best_d = 0.0
    for i, (x1, y1, x2, y2) in enumerate(SKELETON_SEGMENTS):
        t, dist, _, _ = project_point_on_segment(x, y, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_d = (CUMULATIVE_LENGTHS[i] + t * SEGMENT_LENGTHS[i]) / TOTAL_LENGTH
    return best_d


def compute_distance_to_skeleton(x, y):
    """Minimum distance from point (x, y) to the U-maze skeleton."""
    best_dist = np.inf
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        _, dist, _, _ = project_point_on_segment(x, y, x1, y1, x2, y2)
        best_dist = min(best_dist, dist)
    return best_dist


def d_to_zone(d):
    """Convert curvilinear distance d to zone label (0=Left, 1=Top, 2=Right)."""
    if d < D_LEFT_END:
        return 0
    elif d < D_RIGHT_START:
        return 1
    else:
        return 2


def compute_all_geometry(positions):
    """Compute curvilinear distances and zone labels for all positions.

    Args:
        positions: (N, 2) array of (x, y) positions

    Returns:
        curvilinear_d: (N,) float32 array
        zone_labels: (N,) int64 array
    """
    curvilinear_d = np.array([
        compute_curvilinear_distance(x, y) for x, y in positions
    ], dtype=np.float32)
    zone_labels = np.array([d_to_zone(d) for d in curvilinear_d], dtype=np.int64)
    return curvilinear_d, zone_labels
