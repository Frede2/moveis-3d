"""Tileable procedural textures (numpy) for the chair materials."""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

rng = np.random.default_rng(7)


def fft_noise(n, beta, seed):
    """Periodic 1/f^beta noise in [0,1]."""
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, n))
    f = np.fft.fft2(w)
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1
    f *= 1.0 / k ** beta
    f[0, 0] = 0
    out = np.real(np.fft.ifft2(f))
    out -= out.min()
    return out / out.max()


def worley(n, cells, seed):
    """Periodic F1, F2 cellular distances (normalised)."""
    r = np.random.default_rng(seed)
    pts = r.random((cells, 2))
    tiled = np.vstack([pts + [dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    tree = cKDTree(tiled)
    g = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(g, g, indexing="ij")
    d, _ = tree.query(np.c_[X.ravel(), Y.ravel()], k=2)
    s = 1.0 / np.sqrt(cells)
    return (d[:, 0] / s).reshape(n, n), (d[:, 1] / s).reshape(n, n)


def height_to_normal(h, strength):
    dx = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    dy = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(h)
    l = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    n = np.stack([nx / l, ny / l, nz / l], -1)
    # image rows = v, cols = u; glTF expects +Y up (OpenGL convention)
    img = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(np.transpose(img, (1, 0, 2)))


def to_img(a):
    a = np.clip(a, 0, 1)
    return Image.fromarray(np.transpose((a * 255).astype(np.uint8), (1, 0, 2)) if a.ndim == 3 else (a.T * 255).astype(np.uint8))


def mr_image(rough, metal):
    """glTF metallicRoughness: G = roughness, B = metallic."""
    n = rough.shape[0]
    rgb = np.zeros((n, n, 3))
    rgb[..., 1] = rough
    rgb[..., 2] = metal
    return to_img(rgb)


def leather(n=1024):
    from scipy.ndimage import gaussian_filter
    f1, f2 = worley(n, 4200, 11)
    crease = np.clip(f2 - f1, 0, 1)                   # 0 on the cell borders
    pebble = np.sqrt(np.clip(crease / 0.6, 0, 1))     # soft rounded pebbles
    pebble = gaussian_filter(pebble, 1.6, mode="wrap")
    micro = fft_noise(n, 1.0, 3)
    broad = fft_noise(n, 2.2, 5)
    h = 0.80 * pebble + 0.12 * micro + 0.08 * broad
    normal = height_to_normal(h, 3.6)
    rough = 0.50 + 0.08 * (1 - pebble) + 0.04 * (broad - 0.5)
    base = 0.055 + 0.012 * (broad - 0.5) + 0.010 * (pebble - 0.5)
    col = np.stack([base, base, base * 1.06], -1)
    return to_img(col), mr_image(rough, np.zeros_like(rough)), normal


def textured_plastic(n=512):
    stip = fft_noise(n, 0.6, 21)
    mott = fft_noise(n, 2.0, 22)
    h = 0.8 * stip + 0.2 * mott
    normal = height_to_normal(h, 4.0)
    rough = 0.70 + 0.12 * (stip - 0.5) + 0.10 * (mott - 0.5)
    base = 0.075 + 0.030 * (mott - 0.5) + 0.015 * (stip - 0.5)  # slightly dusty, matte
    col = np.stack([base, base * 1.01, base * 1.03], -1)
    return to_img(col), mr_image(rough, np.zeros_like(rough)), normal


def fabric(n=512):
    g = np.arange(n)
    U, V = np.meshgrid(g, g, indexing="ij")
    weave = 0.5 + 0.25 * np.sin(U * 2 * np.pi / 8) * np.sign(np.sin(V * 2 * np.pi / 16)) + \
        0.25 * np.sin(V * 2 * np.pi / 8) * np.sign(np.sin(U * 2 * np.pi / 16))
    fuzz = fft_noise(n, 0.4, 31)
    h = 0.7 * weave + 0.3 * fuzz
    normal = height_to_normal(h, 3.0)
    rough = np.full((n, n), 0.95)
    base = 0.035 + 0.02 * (fuzz - 0.5)
    col = np.stack([base, base, base], -1)
    return to_img(col), mr_image(rough, np.zeros_like(rough)), normal


if __name__ == "__main__":
    for name, fn in [("leather", leather), ("plastic", textured_plastic), ("fabric", fabric)]:
        c, mr, nm = fn()
        c.save(f"tex_{name}_color.png"); mr.save(f"tex_{name}_mr.png"); nm.save(f"tex_{name}_normal.png")
        print(name, c.size)
