"""Detailed office chair, modelled with signed distance fields + marching cubes.
Units: metres, Y up, chair faces +Z. Dimensions estimated (not measured)."""
import numpy as np
import trimesh
import fast_simplification
from skimage.measure import marching_cubes
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
import textures as T

# ---------------------------------------------------------------- SDF helpers
L2 = lambda *a: np.sqrt(sum(x * x for x in a))


def sd_round_box(x, y, z, bx, by, bz, r):
    qx, qy, qz = np.abs(x) - bx + r, np.abs(y) - by + r, np.abs(z) - bz + r
    out = L2(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0))
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def sd_round_rect2(u, v, bu, bv, r):
    qu, qv = np.abs(u) - bu + r, np.abs(v) - bv + r
    return L2(np.maximum(qu, 0), np.maximum(qv, 0)) + np.minimum(np.maximum(qu, qv), 0) - r


def sd_cyl_y(x, y, z, r, y0, y1, rr=0.0):
    """Vertical capped cylinder between y0..y1 with edge rounding rr."""
    h = (y1 - y0) / 2
    dx = L2(x, z) - (r - rr)
    dy = np.abs(y - (y0 + y1) / 2) - (h - rr)
    return L2(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0) - rr


def sd_capsule(x, y, z, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    px, py, pz = x - a[0], y - a[1], z - a[2]
    t = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / (ba @ ba), 0, 1)
    return L2(px - ba[0] * t, py - ba[1] * t, pz - ba[2] * t) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def mesh_from_sdf(fn, lo, hi, h, target_tris):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    xs, ys, zs = [np.linspace(lo[i], lo[i] + h * (n[i] - 1), n[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = fn(X, Y, Z)
    # pad so every shape is closed
    vol = np.pad(vol, 1, constant_values=1.0)
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    v += lo - h
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()
    if len(m.faces) > target_tris:
        red = 1 - target_tris / len(m.faces)
        pv, pf = fast_simplification.simplify(m.vertices, m.faces, red)
        m = trimesh.Trimesh(pv, pf, process=True)
        if m.volume < 0:
            m.invert()
    return m


def lin(*srgb_bytes):
    """sRGB 0-255 -> linear 0-1 floats (glTF baseColorFactor is linear)."""
    return [((c / 255) ** 2.2) for c in srgb_bytes] + [1.0]


# ---------------------------------------------------------------- materials
def mat(name, color, rough, metal=0.0, tex=None):
    kw = dict(name=name, baseColorFactor=[int(c * 255) for c in color] + [255],
              roughnessFactor=rough, metallicFactor=metal)
    if tex:
        c, mr, nm = tex
        c = c.resize((256, 256)); mr = mr.resize((512, 512))
        kw.update(baseColorTexture=c, metallicRoughnessTexture=mr, normalTexture=nm,
                  baseColorFactor=[255, 255, 255, 255], roughnessFactor=1.0, metallicFactor=1.0)
    return PBRMaterial(**kw)


LEATHER = mat("pele", (0.05, 0.05, 0.05), 0.45, tex=T.leather())
PLASTIC = mat("plastico_texturado", (0.07, 0.07, 0.07), 0.8, tex=T.textured_plastic())
FABRIC = mat("tecido", (0.03, 0.03, 0.03), 0.95, tex=T.fabric())
STEEL = PBRMaterial(name="metal_pintado", baseColorFactor=lin(28, 28, 28), metallicFactor=0.0, roughnessFactor=0.6)
CHROME = PBRMaterial(name="cromado", baseColorFactor=lin(240, 242, 245), metallicFactor=1.0, roughnessFactor=0.07)
GREY = PBRMaterial(name="cinza_rodas", baseColorFactor=lin(70, 72, 75), metallicFactor=0.0, roughnessFactor=0.7)
RUBBER = PBRMaterial(name="roda", baseColorFactor=lin(22, 22, 23), metallicFactor=0.0, roughnessFactor=0.62)
SMOOTH = PBRMaterial(name="plastico_liso", baseColorFactor=lin(26, 26, 26), metallicFactor=0.0, roughnessFactor=0.55)

parts = []
GROUP = 'geral'
import os
PART = os.environ.get('PART')


def add(m, material, name, uv_axes=(0, 2), tile=0.12, xf=None):
    """Planar UVs in local coords, then place the mesh in the scene."""
    if uv_axes == "box":
        n = np.abs(m.vertex_normals)
        ax = np.argmax(n, axis=1)
        v = m.vertices
        uv = np.where(ax[:, None] == 0, v[:, [2, 1]], np.where(ax[:, None] == 1, v[:, [0, 2]], v[:, [0, 1]])) / tile
        m.visual = TextureVisuals(uv=uv, material=material)
    elif uv_axes is not None:
        uv = m.vertices[:, list(uv_axes)] / tile
        m.visual = TextureVisuals(uv=uv, material=material)
    else:
        m.visual = TextureVisuals(material=material)
    if xf is not None:
        m.apply_transform(xf)
    parts.append((name, m, GROUP))


def Rx(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0])


def Ry(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])


def Tr(x, y, z):
    return trimesh.transformations.translation_matrix([x, y, z])


# ---------------------------------------------------------------- dimensions
SEAT_TOP = 0.485
SEAT_HX, SEAT_HZ, SEAT_HY = 0.25, 0.24, 0.045
SEAT_CY = SEAT_TOP - SEAT_HY - 0.012           # dome adds ~1.2 cm on top
SEAT_CZ = 0.03

# ---------------------------------------------------------------- seat cushion
GROUP = "assento"
def seat_sdf(x, y, z):
    t = np.clip((y + SEAT_HY) / (2 * SEAT_HY), 0, 1)
    dome = 0.013 * (1 - (x / SEAT_HX) ** 2).clip(0) * (1 - ((z + 0.03) / SEAT_HZ) ** 2).clip(0) * t
    d = sd_round_box(x, y - dome, z, SEAT_HX, SEAT_HY, SEAT_HZ, 0.034)
    # waterfall front roll
    d = smin(d, sd_capsule(x, y, z, (-0.205, -0.004, SEAT_HZ - 0.035), (0.205, -0.004, SEAT_HZ - 0.035), 0.040), 0.02)
    # piping (welt) around the top edge
    ring = L2(sd_round_rect2(x, z, SEAT_HX - 0.006, SEAT_HZ - 0.004, 0.06), y - (SEAT_HY - 0.012)) - 0.0045
    d = np.minimum(d, ring)
    # small crease where the cushion meets the side panel
    groove = L2(sd_round_rect2(x, z, SEAT_HX - 0.0005, SEAT_HZ + 0.001, 0.06), y + 0.012) - 0.0022
    return d


m = mesh_from_sdf(seat_sdf, (-0.27, -0.06, -0.26), (0.27, 0.07, 0.30), 0.0026, 30000)
add(m, LEATHER, "assento", "box", xf=Tr(0, SEAT_CY, SEAT_CZ))

# fabric-covered underside board
m = mesh_from_sdf(lambda x, y, z: sd_round_box(x, y, z, SEAT_HX - 0.012, 0.007, SEAT_HZ - 0.012, 0.006),
                  (-0.26, -0.012, -0.25), (0.26, 0.012, 0.25), 0.004, 3000)
add(m, FABRIC, "tecido_inferior", (0, 2), tile=0.08, xf=Tr(0, SEAT_CY - SEAT_HY - 0.004, SEAT_CZ))

# ---------------------------------------------------------------- backrest
GROUP = "encosto"
BHX, BHY, BHZ = 0.235, 0.29, 0.036


def back_warp(x, y):
    """Forward offset of the panel: wrap-around sides + lumbar bulge."""
    return 0.055 * (x / BHX) ** 2 + 0.022 * np.exp(-((y + 0.13) / 0.10) ** 2)


def back_sdf(x, y, z):
    zq = z - back_warp(x, y)
    d = sd_round_box(x, y, zq, BHX, BHY, BHZ, 0.03) * 0.92
    # front: soft quilted dome
    d -= 0.006 * (1 - (x / BHX) ** 2).clip(0) * (1 - (y / BHY) ** 2).clip(0) * (zq > 0)
    # piping around the front face
    ring = L2(sd_round_rect2(x, y, BHX - 0.004, BHY - 0.004, 0.05), zq - 0.016) - 0.0045
    d = np.minimum(d, ring)
    # rear seams (from the photos): two horizontal seams, one vertical in the lower panel
    yb1 = 0.19 + 0.012 * (x / BHX) ** 2
    yb2 = -0.02 - 0.035 * (1 - (x / BHX) ** 2)
    zb = zq + BHZ
    s1 = L2(y - yb1, zb + 0.003) - 0.0045
    s2 = L2(y - yb2, zb + 0.003) - 0.0045
    s3 = np.where(y < yb2, L2(x, zb + 0.003), 9.0) - 0.0045
    seams = np.minimum(np.minimum(s1, s2), s3)
    seams = np.where(np.abs(x) < BHX - 0.02, seams, 9.0)
    return np.maximum(d, -seams)


m = mesh_from_sdf(back_sdf, (-0.25, -0.30, -0.05), (0.25, 0.30, 0.10), 0.0024, 40000)
BACK_XF = Tr(0, SEAT_TOP + 0.235, -0.262) @ Rx(-9)
add(m, LEATHER, "encosto", "box", xf=BACK_XF)


# ---------------------------------------------------------------- mechanism
GROUP = "mecanismo"
MECH_Y = SEAT_CY - SEAT_HY - 0.04


def mech_sdf(x, y, z):
    plate = sd_round_box(x, y, z, 0.10, 0.006, 0.12, 0.004)
    body = sd_round_box(x, y + 0.022, z - 0.01, 0.062, 0.018, 0.085, 0.01)
    d = smin(plate, body, 0.01)
    for sx in (-1, 1):   # side flanges with bolt heads
        fl = sd_round_box(x - sx * 0.098, y + 0.012, z, 0.004, 0.016, 0.10, 0.003)
        d = np.minimum(d, fl)
        for zz in (-0.06, 0.06):
            d = np.minimum(d, L2(x - sx * 0.104, y + 0.012, z - zz) - 0.0065)
    return d


m = mesh_from_sdf(mech_sdf, (-0.12, -0.05, -0.14), (0.12, 0.015, 0.14), 0.0018, 7000)
add(m, STEEL, "mecanismo", None, xf=Tr(0, MECH_Y, SEAT_CZ))

# tension knob (ribbed cylinder hanging at the front)
def knob_sdf(x, y, z):
    th = np.arctan2(z, x)
    r = 0.026 + 0.0012 * np.cos(18 * th)
    return sd_cyl_y(x, y, z, r, -0.055, 0.0, 0.004)


m = mesh_from_sdf(knob_sdf, (-0.035, -0.065, -0.035), (0.035, 0.01, 0.035), 0.0015, 3500)
add(m, SMOOTH, "botao_tensao", None, xf=Tr(0.03, MECH_Y - 0.02, SEAT_CZ + 0.11))

# height lever with paddle (left side)
def lever_sdf(x, y, z):
    rod = sd_capsule(x, y, z, (0, 0, 0), (-0.09, -0.02, 0.03), 0.0055)
    pad = sd_round_box(x + 0.12, y + 0.026, z - 0.04, 0.035, 0.007, 0.02, 0.006)
    return smin(rod, pad, 0.01)


m = mesh_from_sdf(lever_sdf, (-0.17, -0.045, -0.02), (0.02, 0.02, 0.075), 0.0015, 2500)
add(m, SMOOTH, "alavanca", None, xf=Tr(-0.1, MECH_Y - 0.01, SEAT_CZ + 0.06))

# ---------------------------------------------------------------- gas lift
GROUP = "cilindro"
GAS_TOP = MECH_Y - 0.035
HUB_TOP = 0.118
m = mesh_from_sdf(lambda x, y, z: sd_cyl_y(x, y, z, 0.0145, GAS_TOP - 0.07, GAS_TOP, 0.002),
                  (-0.02, GAS_TOP - 0.08, -0.02), (0.02, GAS_TOP + 0.01, 0.02), 0.0012, 2500)
add(m, CHROME, "pistao_cromado", None)

def cover_sdf(x, y, z):
    t1 = sd_cyl_y(x, y, z, 0.0245, GAS_TOP - 0.135, GAS_TOP - 0.055, 0.003)
    t2 = sd_cyl_y(x, y, z, 0.0275, GAS_TOP - 0.205, GAS_TOP - 0.13, 0.003)
    t3 = sd_cyl_y(x, y, z, 0.031, HUB_TOP - 0.01, GAS_TOP - 0.2, 0.003)
    return np.minimum(np.minimum(t1, t2), t3)


m = mesh_from_sdf(cover_sdf, (-0.035, HUB_TOP - 0.02, -0.035), (0.035, GAS_TOP - 0.045, 0.035), 0.0018, 5000)
add(m, PLASTIC, "cilindro_telescopico", (0, 1), tile=0.1)

# ---------------------------------------------------------------- five-star base
GROUP = "base"
LEG_ANG = [i * 72 + 18 for i in range(5)]
R_TIP = 0.325


def base_sdf(x, y, z):
    hub = sd_cyl_y(x, y, z, 0.060, 0.062, HUB_TOP + 0.004, 0.014)
    d = hub
    for a in LEG_ANG:
        ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
        u = x * sa + z * ca           # along the leg
        v = x * ca - z * sa           # sideways
        t = np.clip(u / R_TIP, 0, 1)
        hw = 0.030 - 0.009 * t        # half width
        hh = 0.025 - 0.006 * t        # half height
        yc = 0.094 - 0.012 * t        # leg dips towards the tip
        top_dome = 0.004 * (1 - (v / hw) ** 2).clip(0)
        e = sd_round_rect2(v, y - yc - top_dome * (y > yc), hw, hh, 0.013)
        e = np.maximum(e, u - R_TIP)
        e = np.maximum(e, 0.02 - u)
        # rounded nose that turns down into the caster socket
        nose = L2(u - R_TIP, v * 1.15, (y - 0.079) * 1.05) - 0.022
        sock = sd_capsule(u, y, v, (R_TIP, 0.080, 0), (R_TIP, 0.060, 0), 0.017)
        leg = smin(smin(e, nose, 0.014), sock, 0.008)
        d = smin(d, leg, 0.026)
    return d


m = mesh_from_sdf(base_sdf, (-0.37, 0.03, -0.37), (0.37, 0.135, 0.37), 0.0026, 48000)
add(m, PLASTIC, "base_estrela", (0, 2), tile=0.1)

# ---------------------------------------------------------------- twin casters
GROUP = "rodas"
WR = 0.028  # wheel radius


def wheel_sdf(x, y, z):
    # wheel axis along x; two wheels either side of a central housing
    d = 9.0
    for sx in (-1, 1):
        xc = sx * 0.0145
        tire = sd_round_box(L2(y, z) - WR / 2, x - xc, 0 * x, WR / 2, 0.0075, 1, 0.004)
        tire = (np.sqrt(np.maximum(L2(y, z) - WR + 0.004, 0) ** 2 + np.maximum(np.abs(x - xc) - 0.0035, 0) ** 2)
                + np.minimum(np.maximum(L2(y, z) - WR + 0.004, np.abs(x - xc) - 0.0035), 0) - 0.004)
        d = np.minimum(d, tire)
    return d


def hub_sdf(x, y, z):
    """Thin grey ring on the outer face of each wheel (as in the photos)."""
    d = 9.0
    for sx in (-1, 1):
        rr = L2(y, z)
        ring = sd_round_rect2(rr - 0.0225, x - sx * 0.0190, 0.0030, 0.0010, 0.0006)
        d = np.minimum(d, ring)
    return d


def housing_sdf(x, y, z):
    # hood over the top half of the wheels + stem into the leg socket
    hood = sd_round_box(x, y - 0.006, z, 0.0075, 0.022, 0.024, 0.006)
    hood = np.maximum(hood, -(y + 0.004))
    stem = sd_cyl_y(x, y, z, 0.0075, 0.01, 0.034, 0.002)
    return smin(hood, stem, 0.006)


w_mesh = mesh_from_sdf(wheel_sdf, (-0.03, -0.03, -0.03), (0.03, 0.03, 0.03), 0.0011, 2600)
h_mesh = mesh_from_sdf(hub_sdf, (-0.03, -0.02, -0.02), (0.03, 0.02, 0.02), 0.0009, 900)
o_mesh = mesh_from_sdf(housing_sdf, (-0.012, -0.01, -0.032), (0.012, 0.04, 0.032), 0.0011, 1500)
for i, a in enumerate(LEG_ANG):
    ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
    tip = np.array([sa * R_TIP, 0, ca * R_TIP])
    swivel = a + 90 + (i * 37) % 60 - 30     # casters point slightly different ways, like real use
    xf = Tr(tip[0], WR, tip[2]) @ Ry(swivel)
    add(w_mesh.copy(), RUBBER, f"rodas_{i+1}", None, xf=xf)
    add(h_mesh.copy(), GREY, f"tampas_{i+1}", None, xf=xf)
    add(o_mesh.copy(), PLASTIC, f"capa_rodizio_{i+1}", (2, 1), tile=0.1, xf=xf)

# ---------------------------------------------------------------- armrests
GROUP = "bracos"
ARM_X = 0.278
PAD_Y = SEAT_TOP + 0.215


def pad_sdf(x, y, z):
    s = 1 + 0.18 * np.clip(z / 0.13, -1, 1)          # wider at the front
    xq = x / s
    yq = y + 0.007 * (z / 0.13) ** 2                 # ends curve slightly down
    d = sd_round_box(xq, yq, z, 0.040, 0.016, 0.13, 0.015) * s
    d -= 0.004 * (1 - (z / 0.13) ** 2).clip(0) * (yq > 0)
    groove = L2(sd_round_rect2(xq, z, 0.040, 0.13, 0.015) * s, yq + 0.003) - 0.0018
    return np.maximum(d, -groove)


def post_sdf(x, y, z):
    t = np.clip((y + 0.0) / 0.30, 0, 1)
    sx, sz = 0.025 - 0.006 * t, 0.029 - 0.007 * t
    post = sd_round_box(x, y - 0.15, z, sx, 0.15, sz, 0.008)
    # T-head under the pad
    head = sd_round_box(x, y - 0.302, z, 0.022, 0.008, 0.06, 0.006)
    d = smin(post, head, 0.01)
    # height-adjust button on the front of the post
    btn = sd_round_box(x, y - 0.255, z - (sz + 0.004), 0.010, 0.012, 0.006, 0.003)
    return np.minimum(d, btn)


pad_m = mesh_from_sdf(pad_sdf, (-0.06, -0.03, -0.15), (0.06, 0.03, 0.15), 0.0018, 7000)
post_m = mesh_from_sdf(post_sdf, (-0.03, -0.01, -0.07), (0.03, 0.32, 0.07), 0.002, 5000)
for side in (-1, 1):
    add(pad_m.copy(), LEATHER, f"apoio_braco_{side}", "box", tile=0.12, xf=Tr(side * ARM_X, PAD_Y, 0.03))
    post_base = PAD_Y - 0.318
    add(post_m.copy(), SMOOTH, f"poste_braco_{side}", None, xf=Tr(side * ARM_X, post_base, 0.0))
    # chrome sleeve + black socket, then flat steel bar under the seat
    m = mesh_from_sdf(lambda x, y, z: sd_cyl_y(x, y, z, 0.0205, 0.0, 0.036, 0.002),
                      (-0.025, -0.005, -0.025), (0.025, 0.04, 0.025), 0.0012, 2000)
    add(m, CHROME, f"anel_cromado_braco_{side}", None, xf=Tr(side * ARM_X, post_base - 0.036, 0))
    m = mesh_from_sdf(lambda x, y, z: smin(sd_cyl_y(x, y, z, 0.026, 0.0, 0.05, 0.006),
                                           sd_round_box(x, y - 0.012, z, 0.03, 0.012, 0.024, 0.006), 0.01),
                      (-0.04, -0.005, -0.035), (0.04, 0.055, 0.035), 0.0015, 2500)
    add(m, SMOOTH, f"encaixe_braco_{side}", None, xf=Tr(side * ARM_X, post_base - 0.086, 0))
    bar_y = post_base - 0.08
    m = mesh_from_sdf(lambda x, y, z: sd_round_box(x, y, z, 0.105, 0.005, 0.025, 0.003),
                      (-0.11, -0.01, -0.03), (0.11, 0.01, 0.03), 0.002, 1200)
    add(m, STEEL, f"barra_braco_{side}", None, xf=Tr(side * 0.17, bar_y, 0))
    # vertical tab from the bar up to the seat board
    m = mesh_from_sdf(lambda x, y, z: sd_round_box(x, y, z, 0.022, 0.03, 0.02, 0.004),
                      (-0.03, -0.035, -0.025), (0.03, 0.035, 0.025), 0.002, 800)
    add(m, STEEL, f"fixacao_braco_{side}", None, xf=Tr(side * 0.09, bar_y + 0.028, 0))

# ---------------------------------------------------------------- export
sel = [(n, m) for n, m, g in parts if not PART or g in PART.split(",")]
scene = trimesh.Scene()
for name, m in sel:
    scene.add_geometry(m, node_name=name, geom_name=name)
if not PART:
    scene.apply_translation([0, -scene.bounds[0][1], 0])
ext = scene.extents
print("L x A x P (m):", np.round(ext, 3), "| partes:", len(parts),
      "| triangulos:", sum(len(m.faces) for _, m in sel))
OUT = f"parte_{PART}.glb" if PART else "cadeira_detalhada.glb"
scene.export(OUT)
print(OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
