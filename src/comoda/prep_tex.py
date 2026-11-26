"""Turn Poly Haven plank-floor scans into seam-free 'glued solid board' textures.
Output (grain along image x): <name>_col.png, <name>_nor.png (GL), <name>_rough.png, all 8-bit, full res."""
import sys
import numpy as np
import OpenEXR
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

Image.MAX_IMAGE_PIXELS = None
SRC = "/home/claude/tex_in"


def exr(path):
    ch = OpenEXR.File(path).channels()
    k = list(ch)[0]
    return ch[k].pixels.astype(np.float32)


def load(name, folder):
    t = f"{SRC}/{folder}/textures/{name}"
    col = np.asarray(Image.open(t + "_diff_4k.jpg").convert("RGB")).astype(np.float32) / 255
    nor = exr(t + "_nor_gl_4k.exr")            # linear, 0..1 encoded
    rough = exr(t + "_rough_4k.exr")
    if rough.ndim == 3:
        rough = rough[..., 0]
    disp = np.asarray(Image.open(t + "_disp_4k.png")).astype(np.float32)
    return col, nor, rough, disp


def rot90(col, nor, rough, disp):
    """Rotate 90 deg CCW so vertical grain becomes horizontal; fix tangent-space normals."""
    col, rough, disp = np.rot90(col), np.rot90(rough), np.rot90(disp)
    nor = np.rot90(nor).copy()
    x = nor[..., 0] * 2 - 1
    y = nor[..., 1] * 2 - 1
    nor[..., 0] = (-y + 1) / 2
    nor[..., 1] = (x + 1) / 2
    return col, nor, rough, disp


def deseam(col, nor, rough, disp, half=9, prom=150):
    prof = disp.mean(1)
    r = prof - median_filter(prof, 81, mode="wrap")
    seams, _ = find_peaks(-r, prominence=prom, distance=120)
    keep = np.ones(len(prof), bool)
    for s in seams:
        keep[max(0, s - half):s + half + 1] = False
    print("seams removed:", len(seams), "rows kept:", keep.sum())
    return col[keep], nor[keep], rough[keep]


def save(prefix, col, nor, rough):
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(prefix + "_col.png")
    Image.fromarray((np.clip(nor, 0, 1) * 255).astype(np.uint8)).save(prefix + "_nor.png")
    Image.fromarray((np.clip(rough, 0, 1) * 255).astype(np.uint8)).save(prefix + "_rough.png")


if __name__ == "__main__":
    out = "/home/claude/comoda2/tex"
    c = load("oak_wood_planks", "oak_wood_planks_4k_blend")
    save(f"{out}/oak", *deseam(*c))
    c = rot90(*load("japanese_cedar_planks", "japanese_cedar_planks_4k_blend"))
    save(f"{out}/cedar", *deseam(*c))
