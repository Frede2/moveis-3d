"""Cómoda v2: same original design (W 80 x D 42 x H 88 cm), now with scanned wood, rounded edges and 5 finishes.
Usage: python3 build_comoda2.py [finish ...]   -> comoda_<finish>.glb"""
import os
import sys
import io
import numpy as np
import trimesh
import trimesh.exchange.gltf as G
import fast_simplification
from skimage.measure import marching_cubes
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

sys.path.insert(0, "/home/claude/comoda")
import wood as W          # image_layer (cherry-blossom painting)
import finishes as F

# ---- JPEG textures at good quality (trimesh re-saves JPEG at q75 by default)
_orig = G._append_image


def _append_image(img, tree, buffer_items, extension_webp):
    if getattr(img, "format", None) != "JPEG":
        return _orig(img, tree, buffer_items, extension_webp)
    q = img.info.get("q", 88)
    with io.BytesIO() as f:
        img.save(f, format="JPEG", quality=q, subsampling=0 if q >= 90 else 2)
        data = f.getvalue()
    idx = G._buffer_append(buffer_items, data)
    tree["images"].append({"bufferView": idx, "mimeType": "image/jpeg"})
    return len(tree["images"]) - 1


G._append_image = _append_image


def jpg(a, q=88):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    im = Image.open(io.BytesIO(b.getvalue()))
    im.info["q"] = q
    return im


def mr_img(r):
    z = np.zeros(r.shape + (3,), np.float32)
    z[..., 1] = r
    return z


def lin(*c):
    return [(x / 255) ** 2.2 for x in c] + [1.0]


# ------------------------------------------------------------------ geometry helpers
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


def rbox(ext, r, centre=(0, 0, 0), grain=0, uv_off=(0, 0), uvfn=None):
    """Box with rounded edges/corners of radius r. UVs projected per original face,
    grain (texture u) along axis `grain`. Returns a Trimesh with analytic normals."""
    b = np.asarray(ext, float) / 2
    r = min(r, *(b * 0.95))
    t = np.array([0.0, 0.12, 0.35, 0.65, 1.0])
    coords = [np.unique(np.r_[-bi + r * t, bi - r * t[::-1]]) for bi in b]
    V, Fc, N, UV = [], [], [], []
    inner = b - r
    for ax in range(3):
        o1, o2 = [a for a in range(3) if a != ax]
        for s in (-1, 1):
            c1, c2 = coords[o1], coords[o2]
            base = len(V)
            for i in c1:
                for j in c2:
                    p = np.zeros(3)
                    p[ax], p[o1], p[o2] = s * b[ax], i, j
                    q = np.clip(p, -inner, inner)
                    d = p - q
                    n = d / np.linalg.norm(d)
                    V.append(q + n * r)
                    N.append(n)
                    flat = p + centre
                    if grain in (o1, o2):
                        u_ax, v_ax = grain, (o2 if grain == o1 else o1)
                    else:
                        u_ax, v_ax = o1, o2
                    UV.append([flat[u_ax] / F.ALONG + uv_off[0], flat[v_ax] / F.ACROSS + uv_off[1]])
            n1, n2 = len(c1), len(c2)
            for i in range(n1 - 1):
                for j in range(n2 - 1):
                    a_, b_, c_, d_ = base + i * n2 + j, base + (i + 1) * n2 + j, base + (i + 1) * n2 + j + 1, base + i * n2 + j + 1
                    Fc += [[a_, b_, c_], [a_, c_, d_]]
    V, N, UV, Fc = np.array(V) + centre, np.array(N), np.array(UV), np.array(Fc)
    # orient triangles outward
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    flip = (fn * N[Fc].mean(1)).sum(1) < 0
    Fc[flip] = Fc[flip][:, ::-1]
    m = trimesh.Trimesh(V, Fc, vertex_normals=N, process=False)
    m._uv = uvfn(V, N) if uvfn else UV
    return m


# ------------------------------------------------------------------ dimensions (unchanged design)
WID, DEP = 0.80, 0.42
LEG_H = 0.18
TOP_T = 0.026
BODY_TOP = 0.88 - TOP_T
SIDE_T = 0.018
ZF = DEP / 2
GAP = 0.003
INNER_W = WID - 2 * SIDE_T - 2 * GAP
Y0, Y1 = LEG_H + 0.018, BODY_TOP
FH = (Y1 - Y0 - 4 * GAP) / 3
FT = 0.019
PITCH = 0.024
DEPTH = 0.0075
SIDE_H = BODY_TOP - LEG_H

geo = []   # (name, mesh, role)

# top: soft 4 mm edges, grain along the width
geo.append(("tampo", rbox([WID + 0.012, TOP_T, DEP + 0.012], 0.004, (0, BODY_TOP + TOP_T / 2, 0), 0, (0.13, 0.21)), "carcass"))
# sides (own painted texture, grain vertical)
for s in (-1, 1):
    def side_uv(v, n, s=s):
        # outer face: painted mapping; edges fold back onto the panel so the grain carries round the corner
        d = np.clip(WID / 2 - np.abs(v[:, 0]), 0, SIDE_T)          # depth into the panel from the outer face
        z = np.where(np.abs(n[:, 2]) > 0.5, np.sign(v[:, 2]) * (DEP / 2 - d), v[:, 2])
        y = np.where(np.abs(n[:, 1]) > 0.5, np.where(v[:, 1] > (BODY_TOP + LEG_H) / 2, BODY_TOP - d, LEG_H + d), v[:, 1])
        return np.c_[0.5 + s * z / DEP, (y - LEG_H) / SIDE_H]
    m = rbox([SIDE_T, SIDE_H, DEP], 0.0025, (s * (WID / 2 - SIDE_T / 2), (BODY_TOP + LEG_H) / 2, 0), uvfn=side_uv)
    geo.append((f"lateral_{'esq' if s < 0 else 'dir'}", m, "side"))
geo.append(("fundo", rbox([WID - 2 * SIDE_T, 0.018, DEP], 0.0015, (0, LEG_H + 0.009, 0), 0, (0.61, 0.47)), "carcass"))
geo.append(("costas", rbox([WID - 2 * SIDE_T, SIDE_H, 0.006], 0.001, (0, (BODY_TOP + LEG_H) / 2, -DEP / 2 + 0.003), 0, (0.3, 0.7)), "carcass"))
m = trimesh.creation.box([WID - 2 * SIDE_T - 0.002, SIDE_H - 0.02, 0.004])
m.apply_translation([0, (BODY_TOP + LEG_H) / 2, ZF - 0.045])
m._uv = None
geo.append(("interior_sombra", m, "dark"))
# drawer runners/dividers seen through the gaps
for i in range(1, 3):
    y = Y0 + i * (FH + GAP) + GAP / 2
    m = rbox([WID - 2 * SIDE_T, 0.016, 0.04], 0.001, (0, y, ZF - 0.04), 0, (0.4, 0.1 * i))
    geo.append((f"travessa_{i}", m, "carcass"))


def make_front():
    nre = int(round(INNER_W / PITCH))
    pitch = INNER_W / nre
    xs = np.linspace(-INNER_W / 2, INNER_W / 2, nre * 14 + 1)
    fr = ((xs + INNER_W / 2) / pitch) % 1.0
    fr[-1] = 1.0
    zr = DEPTH * np.sqrt(np.clip(1 - (2 * fr - 1) ** 2, 0, 1))
    dz = np.gradient(zr, xs)
    zf = FT / 2
    V, Fc, N = [], [], []
    nx = len(xs)
    for yy in (-FH / 2, FH / 2):
        for k in range(nx):
            V.append([xs[k], yy, zf + zr[k]])
            n = np.array([-dz[k], 0, 1.0]); N.append(n / np.linalg.norm(n))
    for k in range(nx - 1):
        a, b, c, d = k, k + 1, nx + k + 1, nx + k
        Fc += [[a, b, c], [a, c, d]]
    for yy, sgn in ((-FH / 2, -1), (FH / 2, 1)):
        base = len(V)
        for k in range(nx):
            V.append([xs[k], yy, zf + zr[k]]); N.append([0, sgn, 0])
            V.append([xs[k], yy, zf - 0.0005]); N.append([0, sgn, 0])
        for k in range(nx - 1):
            p, q, r_, t = base + 2 * k, base + 2 * k + 2, base + 2 * k + 3, base + 2 * k + 1
            Fc += ([[p, t, r_], [p, r_, q]] if sgn > 0 else [[p, q, r_], [p, r_, t]])
    return trimesh.Trimesh(np.array(V), np.array(Fc), vertex_normals=np.array(N), process=False)


face0 = make_front()
FRONT_OFF = [(0.05, 0.08), (0.52, 0.41), (0.27, 0.74)]   # each drawer cut from a different part of the board
for i in range(3):
    yc = Y0 + GAP + FH / 2 + i * (FH + GAP)
    m = face0.copy()
    m.apply_translation([0, yc, ZF - FT / 2])
    ou, ov = FRONT_OFF[i]
    m._uv = np.c_[m.vertices[:, 0] / F.ALONG + ou, m.vertices[:, 1] / F.ACROSS + ov]
    geo.append((f"frente_ripado_{i+1}", m, "front"))
    geo.append((f"frente_tabua_{i+1}", rbox([INNER_W, FH, FT], 0.0015, (0, yc, ZF - FT / 2), 0, (ou, ov)), "front"))


def bar_sdf(x, y, z):
    bar = sd_round_box(x, y, z - 0.032, 0.085, 0.006, 0.0065, 0.0045)
    legs = np.minimum(sd_round_box(x - 0.072, y, z - 0.016, 0.006, 0.0055, 0.016, 0.0035),
                      sd_round_box(x + 0.072, y, z - 0.016, 0.006, 0.0055, 0.016, 0.0035))
    return smin(bar, legs, 0.004)


handle = mesh_from_sdf(bar_sdf, (-0.1, -0.015, -0.002), (0.1, 0.015, 0.045), 0.0006, 9000)
for i in range(3):
    yc = Y0 + GAP + FH / 2 + i * (FH + GAP)
    h = handle.copy()
    h.apply_translation([0, yc + FH * 0.18, ZF + DEPTH - 0.002])
    h._uv = None
    geo.append((f"puxador_{i+1}", h, "metal"))


def leg_sdf(x, y, z):
    t = np.clip(y / LEG_H, 0, 1)
    r = 0.0135 + 0.0085 * t
    body = np.maximum(L2(x, z) - r, np.maximum(-y + 0.012, y - LEG_H))
    return np.maximum(body, -y)


def tip_sdf(x, y, z):
    return np.maximum(L2(x, z) - 0.0142, np.abs(y - 0.007) - 0.0072)


leg = mesh_from_sdf(leg_sdf, (-0.025, 0.0, -0.025), (0.025, LEG_H + 0.003, 0.025), 0.0009, 2500)
tip = mesh_from_sdf(tip_sdf, (-0.018, -0.002, -0.018), (0.018, 0.017, 0.018), 0.0008, 1200)
SPLAY = 5.0
k = 0
for sx in (-1, 1):
    for sz in (-1, 1):
        px, pz = sx * (WID / 2 - 0.055), sz * (DEP / 2 - 0.05)
        xf = trimesh.transformations.translation_matrix([px, LEG_H, pz]) @ \
            trimesh.transformations.rotation_matrix(np.radians(SPLAY), [sz, 0, -sx]) @ \
            trimesh.transformations.translation_matrix([0, -LEG_H, 0])
        m = leg.copy()
        uv = np.c_[m.vertices[:, 1] / F.ALONG + 0.2 * k, np.arctan2(m.vertices[:, 2], m.vertices[:, 0]) * 0.022 / F.ACROSS + 0.3 * k]
        m.apply_transform(xf)
        m._uv = uv
        geo.append((f"pe_{sx}_{sz}", m, "carcass"))
        t_ = tip.copy()
        t_.apply_transform(xf)
        t_._uv = None
        geo.append((f"ponteira_{sx}_{sz}", t_, "rubber"))
        k += 1

# ------------------------------------------------------------------ materials per finish
METALS = {
    "preto": PBRMaterial(name="metal_preto_mate", baseColorFactor=lin(30, 30, 31), metallicFactor=0.8, roughnessFactor=0.42),
    "latao": PBRMaterial(name="latao_escovado", baseColorFactor=lin(201, 160, 92), metallicFactor=1.0, roughnessFactor=0.33),
}
DARK = PBRMaterial(name="interior", baseColorFactor=lin(40, 30, 22), metallicFactor=0.0, roughnessFactor=0.95)
RUBBER = PBRMaterial(name="ponteira", baseColorFactor=lin(24, 24, 24), metallicFactor=0.0, roughnessFactor=0.7)
FLOWER = "/mnt/user-data/uploads/b9efa393-image.png"
SW, SH = 640, 1024
# side crop on the full-res scan: along grain = panel height, across = depth (2048 px per metre)
SIDE_CROP = (1500, 1900, int(SIDE_H * 4096 / F.ALONG), int(DEP * 4096 / F.ALONG))


def wood_material(key, name):
    c, n, r = F.maps(key, size=(2048, 2048))
    r = np.asarray(Image.fromarray((np.clip(r, 0, 1) * 255).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)).astype(np.float32) / 255
    return PBRMaterial(name=name, baseColorTexture=jpg(c, 88), normalTexture=jpg(n, 92),
                       metallicRoughnessTexture=jpg(mr_img(r), 90),
                       baseColorFactor=[255, 255, 255, 255], metallicFactor=1.0, roughnessFactor=1.0)


def side_material(key):
    c, n, r = F.maps(key, crop=SIDE_CROP, size=(SW, SH), rot=True)
    fl = W.image_layer(FLOWER, SW, SH)                       # (SH, SW, 4), top row = top of panel
    a = fl[..., 3:4] * 0.93
    lacquer = key == "salvia"
    wood_l = (c * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    c = c * (1 - a) + fl[..., :3] * a * ((0.97 + 0.03 * wood_l) if lacquer else (0.90 + 0.12 * wood_l))
    r = r * (1 - a[..., 0]) + 0.40 * a[..., 0]
    nf = n * 2 - 1
    nf[..., :2] *= (1 - 0.55 * a)
    nf /= np.linalg.norm(nf, axis=-1, keepdims=True)
    return PBRMaterial(name=f"lateral_flor_{key}", baseColorTexture=jpg(c, 90), normalTexture=jpg(nf * 0.5 + 0.5, 92),
                       metallicRoughnessTexture=jpg(mr_img(r), 90),
                       baseColorFactor=[255, 255, 255, 255], metallicFactor=1.0, roughnessFactor=1.0)


def build(key):
    f = F.FINISHES[key]
    front = wood_material(key, f"madeira_{key}")
    carcass = wood_material(f["carcass"], f"madeira_{f['carcass']}") if f.get("carcass") else front
    mats = {"front": front, "carcass": carcass, "side": side_material(key), "metal": METALS[f["metal"]],
            "dark": DARK, "rubber": RUBBER}
    scene = trimesh.Scene()
    for name, m, role in geo:
        mm = m.copy()
        uv = getattr(m, "_uv", None)
        mm.visual = TextureVisuals(uv=uv, material=mats[role]) if uv is not None else TextureVisuals(material=mats[role])
        scene.add_geometry(mm, node_name=name, geom_name=name)
    scene.apply_translation([0, -scene.bounds[0][1], 0])
    out = f"comoda_{key}.glb"
    scene.export(out)
    print(key, "L x A x P:", np.round(scene.extents, 3), "| tris:", sum(len(g.faces) for g in scene.geometry.values()),
          "|", round(os.path.getsize(out) / 1e6, 2), "MB")


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(F.FINISHES)):
        build(key)
