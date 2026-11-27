"""Bake a bouclé (looped-yarn) fabric into a Meshy texture atlas.
A small tileable height map of yarn loops is projected onto the mesh in world space (triplanar, real scale),
and baked into the atlas: colour (shading between loops), tangent-space normal (blended with Meshy's) and roughness.
Usage: python3 boucle.py scaled.glb out_prefix"""
import sys
import numpy as np
import trimesh
from numba import njit
from PIL import Image
from scipy.ndimage import gaussian_filter

TILE_M = 0.06          # one tile of the pattern = 6 cm of fabric
N = 512
rng = np.random.default_rng(7)


def loop_tile():
    """Tileable height field: hundreds of small overlapping yarn loops (rings) + fibre fuzz."""
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    h = np.zeros((N, N), np.float32)
    for _ in range(int(__import__('os').environ.get('NLOOPS', 320))):
        cx, cy = rng.uniform(0, N, 2)
        r = rng.uniform(30, 50)                 # loop radius in px (~3.5-6 mm): the scale the atlas can actually show
        w = r * rng.uniform(0.40, 0.55)
        dx = (xx - cx + N / 2) % N - N / 2      # wrap -> tileable
        dy = (yy - cy + N / 2) % N - N / 2
        sx = rng.uniform(0.75, 1.3)             # slightly elliptic, random orientation
        a = rng.uniform(0, np.pi)
        u = (dx * np.cos(a) + dy * np.sin(a)) * sx
        v = (-dx * np.sin(a) + dy * np.cos(a)) / sx
        d = np.abs(np.hypot(u, v) - r)
        ring = np.clip(1 - (d / w) ** 2, 0, 1)
        h = np.maximum(h, ring * rng.uniform(0.6, 1.0))
    h = 0.85 * h + 0.15 * gaussian_filter(rng.random((N, N)).astype(np.float32), 1.0, mode="wrap") * 2
    h = gaussian_filter(h, 0.8, mode="wrap")
    h = (h - h.min()) / (h.max() - h.min())
    gy, gx = np.gradient(np.pad(h, 1, mode="wrap"))
    return h, gx[1:-1, 1:-1].astype(np.float32), gy[1:-1, 1:-1].astype(np.float32)


@njit(cache=True)
def samp(img, u, v):
    n = img.shape[0]
    x = (u % 1.0) * n; y = (v % 1.0) * n
    x0 = int(x) % n; y0 = int(y) % n; x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
    fx = x - int(x); fy = y - int(y)
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy


@njit(cache=True)
def bake(uvpx, pos, nrm, faces, H, GX, GY, tile, out_h, out_n, mask, BUMP):
    S = out_h.shape[0]
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        ax, ay = uvpx[a]; bx, by = uvpx[b]; cx, cy = uvpx[c]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        # per-triangle tangent frame from UV derivatives (u -> +x image, v -> up image = -y)
        e1 = pos[b] - pos[a]; e2 = pos[c] - pos[a]
        du1 = (bx - ax); dv1 = -(by - ay); du2 = (cx - ax); dv2 = -(cy - ay)
        r = du1 * dv2 - du2 * dv1
        if abs(r) < 1e-12:
            continue
        T = (e1 * dv2 - e2 * dv1) / r
        B = (e2 * du1 - e1 * du2) / r
        x0 = max(int(min(ax, bx, cx)) - 1, 0); x1 = min(int(max(ax, bx, cx)) + 2, S)
        y0 = max(int(min(ay, by, cy)) - 1, 0); y1 = min(int(max(ay, by, cy)) + 2, S)
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5; py = y + 0.5
                w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                w2 = 1 - w0 - w1
                if w0 < -0.02 or w1 < -0.02 or w2 < -0.02:
                    continue
                p = pos[a] * w0 + pos[b] * w1 + pos[c] * w2
                n = nrm[a] * w0 + nrm[b] * w1 + nrm[c] * w2
                n = n / (np.sqrt((n * n).sum()) + 1e-9)
                wx = abs(n[0]) ** 4; wy = abs(n[1]) ** 4; wz = abs(n[2]) ** 4
                ws = wx + wy + wz
                wx /= ws; wy /= ws; wz /= ws
                # three projections (offset so they don't line up)
                h = 0.0; g = np.zeros(3)
                u = p[1] / tile; v = p[2] / tile
                hx = samp(H, u, v); h += wx * hx
                g[1] += wx * samp(GX, u, v); g[2] += wx * samp(GY, u, v)
                u = p[0] / tile + 0.37; v = p[2] / tile + 0.11
                h += wy * samp(H, u, v)
                g[0] += wy * samp(GX, u, v); g[2] += wy * samp(GY, u, v)
                u = p[0] / tile + 0.71; v = p[1] / tile + 0.53
                h += wz * samp(H, u, v)
                g[0] += wz * samp(GX, u, v); g[1] += wz * samp(GY, u, v)
                g = g - n * (g * n).sum()               # keep the tangential part
                np_ = n - g * BUMP                      # bump strength
                np_ = np_ / np.sqrt((np_ * np_).sum())
                tl = np.sqrt((T * T).sum()) + 1e-9; bl = np.sqrt((B * B).sum()) + 1e-9
                out_n[y, x, 0] = (np_ * T).sum() / tl
                out_n[y, x, 1] = (np_ * B).sum() / bl
                out_n[y, x, 2] = (np_ * n).sum()
                out_h[y, x] = h
                mask[y, x] = 1


def run(src, prefix):
    s = trimesh.load(src)
    g = list(s.geometry.values())[0]
    mat = g.visual.material
    col = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    nor = np.asarray(mat.normalTexture.convert("RGB")).astype(np.float32) / 255
    mr = np.asarray(mat.metallicRoughnessTexture.convert("RGB")).astype(np.float32) / 255
    S = col.shape[0]
    uv = g.visual.uv
    uvpx = np.c_[uv[:, 0] * S, (1 - uv[:, 1]) * S].astype(np.float64)
    H, GX, GY = loop_tile()
    out_h = np.zeros((S, S), np.float32); out_n = np.zeros((S, S, 3), np.float32); mask = np.zeros((S, S), np.uint8)
    bake(uvpx, g.vertices.astype(np.float64), g.vertex_normals.astype(np.float64), g.faces.astype(np.int64),
         H.astype(np.float64), GX.astype(np.float64), GY.astype(np.float64), TILE_M, out_h, out_n, mask, float(__import__('os').environ.get('BUMP', 2.5)))
    m = mask[..., None].astype(np.float32)
    print("texels baked:", mask.mean().round(3))
    # colour: soft shading between loops, keep Meshy's colour and seams
    shade = 0.93 + 0.10 * out_h[..., None]
    col2 = col * (1 - m) + np.clip(col * shade, 0, 1) * m
    # normal: blend (whiteout) bouclé detail with Meshy's normal
    mn = nor * 2 - 1
    bn = out_n
    nx = mn[..., 0] + bn[..., 0]; ny = mn[..., 1] + bn[..., 1]; nz = mn[..., 2] * np.maximum(bn[..., 2], 0.2)
    bl = np.stack([nx, ny, nz], -1); bl /= np.linalg.norm(bl, axis=-1, keepdims=True) + 1e-9
    nor2 = nor * (1 - m) + (bl * 0.5 + 0.5) * m
    # roughness: fabric is matte; loop tops slightly smoother
    r = np.clip(0.92 - 0.10 * out_h, 0.75, 1.0)
    mr2 = mr.copy(); mr2[..., 1] = mr[..., 1] * (1 - m[..., 0]) + r * m[..., 0]; mr2[..., 2] = 0
    for name, arr in (("col", col2), ("nor", nor2), ("mr", mr2)):
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(f"{prefix}_{name}.png")
    Image.fromarray((H * 255).astype(np.uint8)).save(f"{prefix}_tile.png")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
