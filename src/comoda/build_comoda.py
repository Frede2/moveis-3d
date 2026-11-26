"""Original 3-drawer oak chest with fluted fronts and a hand-painted pink flower on the sides.
Units: metres, Y up, front faces +Z. Design and dimensions by us (W 80 x D 42 x H 88 cm)."""
import os
import numpy as np
import trimesh
import fast_simplification
from skimage.measure import marching_cubes
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
import wood as W

L2 = lambda *a: np.sqrt(sum(x * x for x in a))


def sd_round_box(x, y, z, bx, by, bz, r):
    qx, qy, qz = np.abs(x) - bx + r, np.abs(y) - by + r, np.abs(z) - bz + r
    out = L2(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0))
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def mesh_from_sdf(fn, lo, hi, h, target):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    xs, ys, zs = [np.linspace(lo[i], lo[i] + h * (n[i] - 1), n[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.pad(fn(X, Y, Z), 1, constant_values=1.0)
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    m = trimesh.Trimesh(v + lo - h, f, process=True)
    if m.volume < 0:
        m.invert()
    if len(m.faces) > target:
        pv, pf = fast_simplification.simplify(m.vertices, m.faces, 1 - target / len(m.faces))
        m = trimesh.Trimesh(pv, pf, process=True)
        if m.volume < 0:
            m.invert()
    return m


def lin(*c):
    return [(x / 255) ** 2.2 for x in c] + [1.0]


# ------------------------------------------------------------------ textures
col, rough, height = W.oak_maps(1024, 2048, seed=1)
nrm = W.normal_from_height(height, 0.9)
OAK = PBRMaterial(name="carvalho_natural", baseColorTexture=W.img(col).resize((512, 1024)),
                  metallicRoughnessTexture=W.mr(rough).resize((512, 1024)),
                  normalTexture=W.img(nrm), baseColorFactor=[255, 255, 255, 255],
                  metallicFactor=1.0, roughnessFactor=1.0)

# side panel: one non-repeating oak board + the painted flower
SW, SH = 620, 1030
scol, srough, sheight = W.oak_maps(SW, SH, seed=4)
fl = W.image_layer("/mnt/user-data/uploads/b9efa393-image.png", SW, SH)  # (rows=SH, cols=SW, 4)
fl = np.transpose(fl, (1, 0, 2))                  # -> (u, v, 4)
a = fl[..., 3:4] * 0.93
paint = fl[..., :3] ** 1.0
scol = scol * (1 - a) + paint * a * (0.92 + 0.08 * scol)  # grain slightly shows through
srough = srough * (1 - a[..., 0]) + 0.42 * a[..., 0]
sheight = sheight * (1 - 0.6 * a[..., 0]) + a[..., 0] * 0.6
snrm = W.normal_from_height(sheight, 0.9)
SIDE = PBRMaterial(name="lateral_cerejeira", baseColorTexture=W.img(scol),
                   metallicRoughnessTexture=W.mr(srough), normalTexture=W.img(snrm),
                   baseColorFactor=[255, 255, 255, 255], metallicFactor=1.0, roughnessFactor=1.0)

METAL = PBRMaterial(name="metal_preto_mate", baseColorFactor=lin(34, 34, 35), metallicFactor=0.75, roughnessFactor=0.38)
DARK = PBRMaterial(name="interior", baseColorFactor=lin(58, 44, 32), metallicFactor=0.0, roughnessFactor=0.9)
RUBBER = PBRMaterial(name="ponteira", baseColorFactor=lin(24, 24, 24), metallicFactor=0.0, roughnessFactor=0.7)

parts = []
GROUP = "geral"
PART = os.environ.get("PART")


def add(m, mat, name, uv=None):
    if uv is not None:
        m.visual = TextureVisuals(uv=uv(m.vertices), material=mat)
    else:
        m.visual = TextureVisuals(material=mat)
    parts.append((name, m, GROUP))


# ------------------------------------------------------------------ dimensions
WID, DEP = 0.80, 0.42
LEG_H = 0.18
TOP_T = 0.026
BODY_TOP = 0.88 - TOP_T
SIDE_T = 0.018
ZF = DEP / 2            # front plane
TILE_U, TILE_V = 0.22, 0.9  # metres covered by one oak texture tile (across / along grain)

# ------------------------------------------------------------------ top
GROUP = "caixa"
m = trimesh.creation.box([WID + 0.012, TOP_T, DEP + 0.012]).subdivide().subdivide()
m.unmerge_vertices()
m.apply_translation([0, BODY_TOP + TOP_T / 2, 0])
add(m, OAK, "tampo", lambda v: np.c_[v[:, 2] / TILE_U, v[:, 0] / TILE_V])   # grain along the width

# sides with the painted flower (non-tiling UV over the panel)
for s in (-1, 1):
    m = trimesh.creation.box([SIDE_T, BODY_TOP - LEG_H, DEP])
    m = m.subdivide().subdivide()
    m.unmerge_vertices()
    m.apply_translation([s * (WID / 2 - SIDE_T / 2), (BODY_TOP + LEG_H) / 2, 0])
    sign = s
    add(m, SIDE, f"lateral_{'esq' if s < 0 else 'dir'}",
        lambda v, sign=sign: np.c_[(0.5 + sign * v[:, 2] / DEP), (v[:, 1] - LEG_H) / (BODY_TOP - LEG_H)])

# bottom, back panel and dark interior behind the drawer gaps
m = trimesh.creation.box([WID - 2 * SIDE_T, 0.018, DEP])
m.apply_translation([0, LEG_H + 0.009, 0])
add(m, OAK, "fundo", lambda v: np.c_[v[:, 2] / TILE_U, v[:, 0] / TILE_V])
m = trimesh.creation.box([WID - 2 * SIDE_T, BODY_TOP - LEG_H, 0.006])
m.apply_translation([0, (BODY_TOP + LEG_H) / 2, -DEP / 2 + 0.003])
add(m, OAK, "costas", lambda v: np.c_[v[:, 0] / TILE_U, v[:, 1] / TILE_V])
m = trimesh.creation.box([WID - 2 * SIDE_T - 0.002, BODY_TOP - LEG_H - 0.02, 0.004])
m.apply_translation([0, (BODY_TOP + LEG_H) / 2, ZF - 0.045])
add(m, DARK, "interior_sombra")

# ------------------------------------------------------------------ fluted drawer fronts
GROUP = "gavetas"
GAP = 0.003
INNER_W = WID - 2 * SIDE_T - 2 * GAP
Y0, Y1 = LEG_H + 0.018, BODY_TOP
FH = (Y1 - Y0 - 4 * GAP) / 3
FT = 0.019
PITCH = 0.024
DEPTH = 0.0075


def make_front():
    """Drawer front: solid board + exact semicircular reeds on the face (local origin = centre)."""
    nre = int(round(INNER_W / PITCH))
    pitch = INNER_W / nre
    xs = np.linspace(-INNER_W / 2, INNER_W / 2, nre * 14 + 1)
    fr = ((xs + INNER_W / 2) / pitch) % 1.0
    fr[-1] = 1.0
    zr = DEPTH * np.sqrt(np.clip(1 - (2 * fr - 1) ** 2, 0, 1))
    dz = np.gradient(zr, xs)
    zf = FT / 2
    V, F, N, UV = [], [], [], []
    nx = len(xs)
    # reeded face: two rows (bottom, top), reeds are straight in y
    for yy in (-FH / 2, FH / 2):
        for k in range(nx):
            V.append([xs[k], yy, zf + zr[k]])
            n = np.array([-dz[k], 0, 1.0]); N.append(n / np.linalg.norm(n))
    for k in range(nx - 1):
        a, b, c, d = k, k + 1, nx + k + 1, nx + k
        F += [[a, b, c], [a, c, d]]
    # top and bottom caps between the reed profile and the board face
    for yy, sgn in ((-FH / 2, -1), (FH / 2, 1)):
        base = len(V)
        for k in range(nx):
            V.append([xs[k], yy, zf + zr[k]]); N.append([0, sgn, 0])
            V.append([xs[k], yy, zf - 0.0005]); N.append([0, sgn, 0])
        for k in range(nx - 1):
            p, q, r_, t = base + 2 * k, base + 2 * k + 2, base + 2 * k + 3, base + 2 * k + 1
            F += ([[p, t, r_], [p, r_, q]] if sgn > 0 else [[p, q, r_], [p, r_, t]])
    V, N = np.array(V), np.array(N)
    face = trimesh.Trimesh(V, np.array(F), vertex_normals=N, process=False)
    board = trimesh.creation.box([INNER_W, FH, FT])
    board = board.subdivide()
    board.unmerge_vertices()
    return face, board


def bar_sdf(x, y, z):
    bar = sd_round_box(x, y, z - 0.032, 0.085, 0.006, 0.0065, 0.0045)
    legs = np.minimum(sd_round_box(x - 0.072, y, z - 0.016, 0.006, 0.0055, 0.016, 0.0035),
                      sd_round_box(x + 0.072, y, z - 0.016, 0.006, 0.0055, 0.016, 0.0035))
    return smin(bar, legs, 0.004)


handle = mesh_from_sdf(bar_sdf, (-0.1, -0.015, -0.002), (0.1, 0.015, 0.045), 0.0006, 9000)
front_face, front_board = make_front()
for i in range(3):
    yc = Y0 + GAP + FH / 2 + i * (FH + GAP)
    for nm, src in (("ripado", front_face), ("tabua", front_board)):
        m = src.copy()
        m.apply_translation([0, yc, ZF - FT / 2])
        add(m, OAK, f"frente_{nm}_{i+1}", lambda v, i=i: np.c_[v[:, 0] / TILE_U + i * 0.37, v[:, 1] / TILE_V])
    h = handle.copy()
    h.apply_translation([0, yc + FH * 0.18, ZF + DEPTH - 0.002])
    add(h, METAL, f"puxador_{i+1}")

# ------------------------------------------------------------------ tapered, splayed legs
GROUP = "pes"


def leg_sdf(x, y, z):
    t = np.clip(y / LEG_H, 0, 1)           # 0 at floor, 1 at top
    r = 0.0135 + 0.0085 * t
    body = np.maximum(L2(x, z) - r, np.maximum(-y + 0.012, y - LEG_H))
    body = np.maximum(body, -y)
    return body


def tip_sdf(x, y, z):
    return sd_round_box(x, y - 0.007, z, 0.0145, 0.0075, 0.0145, 0.006) * 0 + \
        np.maximum(L2(x, z) - 0.0142, np.abs(y - 0.007) - 0.0072)


leg = mesh_from_sdf(leg_sdf, (-0.025, 0.0, -0.025), (0.025, LEG_H + 0.003, 0.025), 0.0009, 2500)
tip = mesh_from_sdf(tip_sdf, (-0.018, -0.002, -0.018), (0.018, 0.017, 0.018), 0.0008, 1200)
SPLAY = 5.0
for sx in (-1, 1):
    for sz in (-1, 1):
        px, pz = sx * (WID / 2 - 0.055), sz * (DEP / 2 - 0.05)
        xf = trimesh.transformations.translation_matrix([px, LEG_H, pz]) @ \
            trimesh.transformations.rotation_matrix(np.radians(SPLAY), [sz, 0, -sx]) @ \
            trimesh.transformations.translation_matrix([0, -LEG_H, 0])
        m = leg.copy()
        uvf = lambda v: np.c_[np.arctan2(v[:, 2], v[:, 0]) * 0.02 / TILE_U, v[:, 1] / TILE_V]
        m.visual = TextureVisuals(uv=uvf(m.vertices), material=OAK)
        m.apply_transform(xf)
        parts.append((f"pe_{sx}_{sz}", m, GROUP))
        t_ = tip.copy()
        t_.visual = TextureVisuals(material=RUBBER)
        t_.apply_transform(xf)
        parts.append((f"ponteira_{sx}_{sz}", t_, GROUP))

# ------------------------------------------------------------------ export
sel = [(n, m) for n, m, g in parts if not PART or g in PART.split(",")]
scene = trimesh.Scene()
for n, m in sel:
    scene.add_geometry(m, node_name=n, geom_name=n)
if not PART:
    scene.apply_translation([0, -scene.bounds[0][1], 0])
out = f"parte_{PART}.glb" if PART else "comoda_flor_rosa.glb"
scene.export(out)
print("L x A x P:", np.round(scene.extents, 3), "| tris:", sum(len(m.faces) for _, m in sel),
      "|", out, round(os.path.getsize(out) / 1e6, 2), "MB")
