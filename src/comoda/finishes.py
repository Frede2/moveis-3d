"""Finishes built from real scanned wood (Poly Haven, CC0): colour grading, roughness and normal strength."""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None
T = "/home/claude/comoda2/tex"
_cache = {}

# physical size of one texture tile (metres): along grain (image x) / across grain (image y)
ALONG, ACROSS = 1.7, 1.7 * 3850 / 4096


def scan(name):
    if name not in _cache:
        col = np.asarray(Image.open(f"{T}/{name}_col.png")).astype(np.float32) / 255
        nor = np.asarray(Image.open(f"{T}/{name}_nor.png")).astype(np.float32) / 255
        rough = np.asarray(Image.open(f"{T}/{name}_rough.png")).astype(np.float32) / 255
        if name == "cedar":
            # even out board-to-board contrast (siding scan is too patchy for furniture)
            low = np.stack([gaussian_filter(col[..., c], 120) for c in range(3)], -1)
            col = col / np.maximum(low, 1e-3) * (0.45 * low + 0.55 * low.mean((0, 1)))
        _cache[name] = (col, nor, rough)
    return _cache[name]


def lum(c):
    return (c * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)


def flatten_normal(nor, k):
    v = nor * 2 - 1
    v[..., :2] *= k
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v * 0.5 + 0.5


# each finish: (scan, colour fn, roughness fn, normal strength, handle)
def g_natural(c):
    return np.clip(c * [1.07, 1.05, 1.02], 0, 1)


def g_smoked(c):
    l = lum(c)
    c = c * 0.75 + l * 0.25
    return np.clip(c ** 1.65 * [0.93, 0.86, 0.80], 0, 1)


def g_nordic(c):
    l = lum(c)
    c = c * 0.5 + l * 0.5                         # less saturated
    c = 1 - (1 - c) * 0.58                        # white-wash, pores stay darker
    return np.clip(c * [1.0, 0.965, 0.915], 0, 1)


def g_cherry(c):
    l = lum(c)
    c = c * 0.75 + l * 0.25
    return np.clip(c ** 1.3 * [0.95, 0.70, 0.60], 0, 1)


SAGE = np.array([0.585, 0.640, 0.565])


def g_sage(c):
    l = lum(c)
    return np.clip(SAGE * (0.965 + 0.10 * (l - l.mean())), 0, 1)


FINISHES = {
    "natural":   dict(nome="Carvalho natural", scan="oak", col=g_natural, rough=lambda r: r * 1.05, nstr=1.0, metal="preto"),
    "fumado":    dict(nome="Carvalho fumado", scan="oak", col=g_smoked, rough=lambda r: r * 0.95, nstr=1.0, metal="latao"),
    "nordico":   dict(nome="Carvalho nórdico", scan="oak", col=g_nordic, rough=lambda r: r * 1.25, nstr=0.85, metal="preto"),
    "cerejeira": dict(nome="Tom cerejeira", scan="cedar", col=g_cherry, rough=lambda r: r * 0.9, nstr=1.0, metal="latao"),
    "salvia":    dict(nome="Verde-sálvia + carvalho", scan="oak", col=g_sage, rough=lambda r: 0.34 + 0.08 * (r - r.mean()), nstr=0.22,
                      metal="latao", carcass="natural"),
}


def maps(key, crop=None, size=None, rot=False):
    """Graded colour / normal / roughness for a finish, optionally cropped (x0, y0, w, h) and rotated."""
    f = FINISHES[key]
    col, nor, rough = scan(f["scan"])
    if crop:
        x0, y0, w, h = crop
        col, nor, rough = col[y0:y0 + h, x0:x0 + w], nor[y0:y0 + h, x0:x0 + w], rough[y0:y0 + h, x0:x0 + w]
    if rot:   # grain vertical: rotate 90 deg CCW and rotate the tangent-space normals with it
        col, rough = np.rot90(col), np.rot90(rough)
        nor = np.rot90(nor).copy()
        x, y = nor[..., 0] * 2 - 1, nor[..., 1] * 2 - 1
        nor[..., 0], nor[..., 1] = (-y + 1) / 2, (x + 1) / 2
    c = f["col"](col)
    r = np.clip(f["rough"](rough), 0.18, 0.95)
    n = flatten_normal(nor.copy(), f["nstr"])
    if size:
        rs = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize(size, Image.LANCZOS)).astype(np.float32) / 255
        c, n, r = rs(c), rs(n), rs(r)
    return c, n, r
