"""Small geometry helpers working in projected metres."""
import numpy as np


def point_segment(P, A, B):
    """Distance from points P to segments A->B (all arrays of shape (n, 2)).

    Returns (distance, s, seg_len) where s is the position of the projection
    along the segment in metres (NOT clipped, so it can be <0 or >len).
    """
    AB = B - A
    seg_len = np.hypot(AB[:, 0], AB[:, 1])
    safe = np.where(seg_len < 1e-6, 1.0, seg_len)
    ux, uy = AB[:, 0] / safe, AB[:, 1] / safe
    ap = P - A
    s = ap[:, 0] * ux + ap[:, 1] * uy
    s_clip = np.clip(s, 0, seg_len)
    cx = A[:, 0] + s_clip * ux
    cy = A[:, 1] + s_clip * uy
    dist = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    return dist, s, seg_len


def signed_lateral(P, A, B):
    """Signed side offset of P from the line A->B (positive = left of travel)."""
    AB = B - A
    L = np.hypot(AB[:, 0], AB[:, 1])
    L = np.where(L < 1e-6, 1.0, L)
    ap = P - A
    return (AB[:, 0] * ap[:, 1] - AB[:, 1] * ap[:, 0]) / L
