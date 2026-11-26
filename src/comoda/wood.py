"""Procedural natural-oak textures and an original hand-painted pink flower."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter


def aniso_noise(n, m, beta, stretch, seed):
    """Periodic noise, elongated along axis 1 (the grain direction). Shape (n, m)."""
    r = np.random.default_rng(seed)
    f = np.fft.fft2(r.standard_normal((n, m)))
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(m)[None, :] * stretch
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1
    f /= k ** beta
    f[0, 0] = 0
    o = np.real(np.fft.ifft2(f))
    o -= o.min()
    return o / o.max()


def oak_maps(n=1024, m=1024, seed=1):
    """Returns float arrays (n, m): colour rgb (sRGB 0-1), roughness, height.
    Axis 0 = across the grain (u), axis 1 = along the grain (v)."""
    broad = aniso_noise(n, m, 1.6, 10, seed)
    lines = aniso_noise(n, m, 1.0, 40, seed + 1)
    # growth rings: warped stripes across u
    u = np.linspace(0, 1, n, endpoint=False)[:, None]
    v = np.linspace(0, 1, m, endpoint=False)[None, :]
    warp = aniso_noise(n, m, 2.0, 3, seed + 2)
    cath = 0.10 * np.cos(2 * np.pi * v) * np.exp(-((u - 0.5) / 0.35) ** 2)
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 16 + warp * 4.5 + cath * 6))
    rings = gaussian_filter(rings ** 4, (1.2, 2.5), mode="wrap")
    # pores: short dark dashes along the grain
    r = np.random.default_rng(seed + 3)
    pores = np.zeros((n, m))
    cnt = int(n * m * 0.0008)
    pu = r.integers(0, n, cnt)
    pv = r.integers(0, m, cnt)
    ln = r.integers(4, 18, cnt)
    for a, b, l in zip(pu, pv, ln):
        idx = (b + np.arange(l)) % m
        pores[a, idx] = 1.0
    pores = gaussian_filter(pores, (0.6, 1.2), mode="wrap")
    pores = np.clip(pores * 2.5, 0, 1)
    # medullary ray flecks (oak's silvery flakes), sparse, across the grain
    rays = aniso_noise(n, m, 1.2, 0.25, seed + 4)
    rays = np.clip((rays - 0.72) * 6, 0, 1) * 0.5

    tone = 0.50 * broad + 0.22 * lines + 0.28 * rings
    light = np.array([0.80, 0.66, 0.47])
    dark = np.array([0.57, 0.41, 0.25])
    col = light[None, None, :] * (1 - tone[..., None]) + dark[None, None, :] * tone[..., None]
    col *= (1 - 0.16 * pores[..., None])
    col += rays[..., None] * np.array([0.05, 0.045, 0.03])
    rough = 0.58 + 0.10 * tone + 0.15 * pores - 0.08 * rays
    height = 0.6 * (1 - pores) + 0.25 * (1 - lines) + 0.15 * rings
    return np.clip(col, 0, 1), np.clip(rough, 0, 1), height


def normal_from_height(h, s):
    dx = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    dy = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    nx, ny, nz = -dx * s, -dy * s, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + 1)
    return np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5


def img(a):
    """(u, v[, c]) array -> PIL image with u = x (columns), v = y (rows, top = v 0)."""
    a = np.clip(a, 0, 1)
    if a.ndim == 2:
        return Image.fromarray((a.T * 255).astype(np.uint8))
    return Image.fromarray((np.transpose(a, (1, 0, 2)) * 255).astype(np.uint8))


def mr(rough, metal=0.0):
    z = np.zeros(rough.shape + (3,))
    z[..., 1] = rough
    z[..., 2] = metal
    return img(z)


def flower_layer(w, h, seed=9):
    """Original pink flower, painted look: RGBA image, transparent background."""
    r = np.random.default_rng(seed)
    S = 3  # supersample for smooth edges
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = w * S * 0.52, h * S * 0.40
    R = min(w, h) * S * 0.20
    # stem: gentle S-curve
    pts = []
    for t in np.linspace(0, 1, 60):
        x = cx + np.sin(t * 3.0) * R * 0.25 - t * R * 0.15
        y = cy + R * 0.6 + t * (h * S * 0.92 - cy - R * 0.6)
        pts.append((x, y))
    d.line(pts, fill=(78, 112, 66, 255), width=int(R * 0.07))
    # two leaves
    for side, ty in ((-1, 0.45), (1, 0.68)):
        px, py = pts[int(ty * 59)]
        leaf = []
        for a in np.linspace(0, np.pi, 30):
            lx = px + side * (a / np.pi * R * 0.9)
            ly = py - np.sin(a) * R * 0.22 - a / np.pi * R * 0.35
            leaf.append((lx, ly))
        for a in np.linspace(np.pi, 0, 30):
            lx = px + side * (a / np.pi * R * 0.85)
            ly = py - a / np.pi * R * 0.35 + np.sin(a) * R * 0.14
            leaf.append((lx, ly))
        d.polygon(leaf, fill=(96, 138, 78, 235))
        d.line([leaf[0], leaf[29]], fill=(70, 104, 60, 200), width=int(R * 0.02))
    # petals: rounded, layered from dark (base) to light (tip) like brush strokes
    def petal(cx0, cy0, a, L, W, c_dark, c_light, alpha=235):
        for k, f in enumerate(np.linspace(1.0, 0.55, 6)):
            c = tuple(int(c_dark[q] + (c_light[q] - c_dark[q]) * k / 5) for q in range(3))
            poly = []
            for t in np.linspace(0, 1, 36):
                s_ = t
                wdt = (np.sin(np.pi * s_) ** 0.55) * W / 2 * f
                poly.append((s_ * L * (0.35 + 0.65 * f), wdt))
            for t in np.linspace(1, 0, 36):
                s_ = t
                wdt = -(np.sin(np.pi * s_) ** 0.55) * W / 2 * f
                poly.append((s_ * L * (0.35 + 0.65 * f), wdt))
            off = L * (1 - f) * 0.45
            P = [(cx0 + (x + off) * np.cos(a) - y * np.sin(a), cy0 + (x + off) * np.sin(a) + y * np.cos(a)) for x, y in poly]
            d.polygon(P, fill=c + (alpha,))
    dark, mid, light = (206, 92, 136), (232, 134, 168), (250, 206, 220)
    off = r.random() * np.pi
    for i in range(6):   # back ring
        a = off + i * np.pi / 3 + r.normal(0, 0.06)
        petal(cx, cy, a, R * r.uniform(0.95, 1.05), R * 0.78, dark, mid)
    for i in range(5):   # front ring, rotated
        a = off + np.pi / 6 + i * 2 * np.pi / 5 + r.normal(0, 0.08)
        petal(cx, cy, a, R * r.uniform(0.62, 0.72), R * 0.62, mid, light)
    # a small bud on a side branch
    bx, by = pts[18][0] + R * 0.55, pts[18][1] - R * 0.25
    d.line([pts[22], (bx, by)], fill=(78, 112, 66, 255), width=int(R * 0.05))
    for k in range(3):
        petal(bx, by, -np.pi / 2 + (k - 1) * 0.45, R * 0.42, R * 0.30, dark, mid)
    # centre
    d.ellipse([cx - R * 0.17, cy - R * 0.17, cx + R * 0.17, cy + R * 0.17], fill=(232, 178, 72, 255))
    for i in range(28):
        a = r.random() * 2 * np.pi
        rr = r.random() ** 0.5 * R * 0.14
        x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
        d.ellipse([x - R * 0.018, y - R * 0.018, x + R * 0.018, y + R * 0.018], fill=(176, 110, 40, 255))
    im = im.filter(ImageFilter.GaussianBlur(S * 0.6)).resize((w, h), Image.LANCZOS)
    # painted-on-wood look: let the grain show through a little
    a = np.asarray(im).astype(float) / 255
    return a


if __name__ == "__main__":
    col, rough, h = oak_maps(512, 512)
    img(col).save("oak_check.png")
    f = flower_layer(420, 700)
    Image.fromarray((f * 255).astype(np.uint8)).save("flower_check.png")
    print("ok")


def image_layer(path, w, h, margin=0.04):
    """Turn a picture on a white background into a paint layer (RGBA 0-1, shape (h, w, 4)).
    White becomes transparent ('unmultiply white'), so ink and petals sit on the wood."""
    src = Image.open(path).convert("RGB")
    sw, sh = src.size
    scale = min(w * (1 - margin) / sw, h * (1 - margin) / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    src = src.resize((nw, nh), Image.LANCZOS)
    c = np.asarray(src).astype(float) / 255
    a0 = np.clip(1 - c.min(axis=2), 0, 1)
    rgb = c                                        # keep the original soft pinks
    a = np.clip(a0 * 2.4, 0, 1) ** 0.9             # opaque where painted, soft at the edges
    a[a0 < 0.03] = 0
    out = np.zeros((h, w, 4))
    x0 = (w - nw) // 2
    y0 = int(h * margin / 2)                        # branch hangs from the top edge
    out[y0:y0 + nh, x0:x0 + nw, :3] = rgb
    out[y0:y0 + nh, x0:x0 + nw, 3] = a
    return out
