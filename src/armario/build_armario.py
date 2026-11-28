"""Roupeiro de pinho maciço, 2 portas almofadadas: 104 x 62 x 203 cm (pés 11 cm, portas 51,5 x 192).
Feito por código (peça geométrica): estrutura de pilares, laterais em quadro com tábuas, portas com
travessas e tábuas verticais, puxadores de madeira. Textura: Poly Haven 'coated_pine_02' (CC0) recolorida
para pinho cru. Exporta versão fechada e aberta (portas a 100°, prateleiras e varão)."""
import io, os, re, sys, types
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

TILE = float(__import__('os').environ.get('TILE', 2.0))                                  # metres per texture tile (both axes)
F = types.SimpleNamespace(ALONG=TILE, ACROSS=TILE)
src = open("/home/claude/comoda2/build_comoda2.py").read()
exec(src[src.index("def rbox"):src.index("# ------------------------------------------------------------------ dimensions")])


def box(ext, centre, grain, off=(0, 0), r=0.0015):
    return rbox(ext, r, centre, grain, off)


# ---------- texture: raw pine from the stained scan
T = "/home/claude/armario/tex"
c = np.asarray(Image.open(f"{T}/hinoki_diff.jpg")).astype(np.float32) / 255
# hinoki scan is already a clean light softwood; warm it towards raw pine
mean = c.reshape(-1, 3).mean(0)
col = np.clip(c / mean * np.array([0.90, 0.79, 0.60]), 0, 1)
nor = Image.open(f"{T}/hinoki_nor_gl.jpg").convert("RGB")
rough = np.asarray(Image.open(f"{T}/hinoki_rough.jpg").convert("L")).astype(np.float32) / 255
rough = np.clip(0.55 + 0.35 * (rough - rough.mean()), 0.4, 0.85)   # raw (unvarnished) wood: fairly matte
mr = np.zeros(rough.shape + (3,), np.float32); mr[..., 1] = rough


def jpg(a, q):
    im = a if isinstance(a, Image.Image) else Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); b.seek(0)
    o = Image.open(b); o.load(); o.format = "JPEG"; return o


PINE = PBRMaterial(name="pinho", baseColorTexture=jpg(col, 88), normalTexture=jpg(nor, 90),
                   metallicRoughnessTexture=jpg(Image.fromarray((mr * 255).astype(np.uint8)).resize((1024, 1024)), 88),
                   metallicFactor=0.0, roughnessFactor=1.0)
SHADOW = PBRMaterial(name="interior_sombra", baseColorFactor=[0.32, 0.25, 0.17, 1], metallicFactor=0.0, roughnessFactor=0.9)
METAL = PBRMaterial(name="varao", baseColorFactor=[0.8, 0.8, 0.82, 1], metallicFactor=1.0, roughnessFactor=0.3)

# ---------- dimensions
W, D, H, LEG = 1.04, 0.62, 2.03, 0.11
DT = 0.022                       # door thickness
ZF = D / 2 - DT                  # carcass front plane (doors sit in front of it)
ZB = -D / 2
P = 0.045                        # corner post section
parts = []
rng = np.random.default_rng(5)
off = lambda: tuple(rng.uniform(0, 1, 2))

# corner posts (they are also the feet)
for sx in (-1, 1):
    for z in (ZF - P / 2, ZB + P / 2):
        parts.append(("pilar", box([P, H, P], (sx * (W / 2 - P / 2), H / 2, z), 1, off(), 0.003), PINE))
# side panels: frame rails + vertical planks set back 6 mm
for sx in (-1, 1):
    x = sx * (W / 2 - P / 2)
    zl = ZF - P; zr = ZB + P; dz = zl - zr; zc = (zl + zr) / 2
    for y, h in ((LEG + 0.04, 0.08), (H - 0.04, 0.08), ((LEG + H) / 2, 0.075)):
        parts.append(("travessa_lateral", box([0.030, h, dz], (x, y, zc), 2, off()), PINE))
    for y0, y1 in ((LEG + 0.08, (LEG + H) / 2 - 0.0375), ((LEG + H) / 2 + 0.0375, H - 0.08)):
        parts.append(("almofada_lateral", box([0.012, y1 - y0, dz], (x - sx * 0.002, (y0 + y1) / 2, zc), 1, off(), 0.0008), PINE))
# top, bottom, back
parts.append(("tampo", box([W, 0.022, D - DT], (0, H - 0.011, (ZF + ZB) / 2), 0, off(), 0.002), PINE))
parts.append(("fundo", box([W - 2 * P, 0.020, D - DT - 0.01], (0, LEG + 0.01, (ZF + ZB) / 2), 0, off()), PINE))
parts.append(("costas", box([W - 2 * P, H - LEG - 0.02, 0.010], (0, (H + LEG) / 2, ZB + 0.012), 1, off(), 0.0008), PINE))
# interior: centre divider, shelves on the right, shelf + rail on the left
yi0, yi1 = LEG + 0.02, H - 0.022
iw = W - 2 * P
parts.append(("divisoria", box([0.018, yi1 - yi0, D - DT - 0.03], (0, (yi0 + yi1) / 2, (ZF + ZB) / 2 - 0.005), 1, off()), PINE))
for y in (yi0 + 0.27, yi0 + 0.635, yi0 + 1.395, yi0 + 1.76):
    parts.append(("prateleira", box([iw / 2 - 0.009, 0.018, D - DT - 0.04], (iw / 4 + 0.0045, y, (ZF + ZB) / 2), 0, off()), PINE))
for y in (yi0 + 0.27, yi0 + 1.76):
    parts.append(("prateleira", box([iw / 2 - 0.009, 0.018, D - DT - 0.04], (-iw / 4 - 0.0045, y, (ZF + ZB) / 2), 0, off()), PINE))
rail = trimesh.creation.cylinder(radius=0.012, height=iw / 2 - 0.02, sections=24)
rail.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
rail.apply_translation([-iw / 4, yi0 + 1.70, (ZF + ZB) / 2]); rail._uv = None
parts.append(("varao", rail, METAL))
shadow = trimesh.creation.box([iw - 0.01, yi1 - yi0 - 0.01, 0.002]); shadow.apply_translation([0, (yi0 + yi1) / 2, ZB + 0.018]); shadow._uv = None
parts.append(("fundo_sombra", shadow, SHADOW))

# ---------- doors (built around their hinge axis so they can be opened)
DW, DH = 0.515, 1.92
ST, RL = 0.070, 0.075                 # stile width, rail height
MID = [1.36 - LEG, 0.76 - LEG]        # centre heights of the two middle rails (from the bottom of the door)


def door(side):
    """side=-1 left door (hinge on the left), +1 right door. Local frame: hinge at x=0, front face at z=DT."""
    g = []
    s = -side                                    # door extends from the hinge towards the centre
    cx = lambda a, b: s * (a + b) / 2            # x centre of a span [a,b] measured from the hinge
    # stiles
    g.append(box([ST, DH, DT], (cx(0, ST), DH / 2, DT / 2), 1, off()))
    g.append(box([ST, DH, DT], (cx(DW - ST, DW), DH / 2, DT / 2), 1, off()))
    # rails
    rails = [RL / 2, DH - RL / 2] + MID
    for yc in rails:
        g.append(box([DW - 2 * ST, RL, DT - 0.002], (cx(ST, DW - ST), yc, DT / 2 - 0.001), 0, off()))
    # plank panels between the rails (vertical V-groove boards = the plank texture turned upright)
    ys = sorted(rails)
    for a, b in zip(ys[:-1], ys[1:]):
        y0, y1 = a + RL / 2, b - RL / 2
        g.append(box([DW - 2 * ST, y1 - y0, 0.012], (cx(ST, DW - ST), (y0 + y1) / 2, DT / 2 - 0.004), 1, off(), 0.0006))
    # wooden pull, vertical, near the meeting edge
    g.append(box([0.020, 0.075, 0.022], (cx(DW - ST / 2 - 0.012, DW - ST / 2 + 0.008), 0.95, DT + 0.011), 1, off(), 0.003))
    return g


def assemble(open_deg):
    sc = trimesh.Scene()
    i = 0
    for name, m, mat in parts:
        mm = m.copy(); uv = getattr(m, "_uv", None)
        mm.visual = TextureVisuals(uv=uv, material=mat) if uv is not None else TextureVisuals(material=mat)
        sc.add_geometry(mm, node_name=f"{name}_{i}", geom_name=f"{name}_{i}"); i += 1
    for side in (-1, 1):
        hinge = np.array([side * W / 2, LEG, ZF])
        ang = np.radians(open_deg) * side
        Tm = trimesh.transformations.translation_matrix(hinge) @ trimesh.transformations.rotation_matrix(ang, [0, 1, 0])
        for j, m in enumerate(door(side)):
            mm = m.copy(); uv = m._uv
            mm.apply_transform(Tm)
            mm.visual = TextureVisuals(uv=uv, material=PINE)
            sc.add_geometry(mm, node_name=f"porta_{'esq' if side < 0 else 'dir'}_{j}", geom_name=f"porta_{side}_{j}")
    return sc


if __name__ == "__main__":
    for deg, out in ((0, "armario.glb"), (100, "armario_aberto.glb")):
        sc = assemble(deg)
        sc.export(out)
        print(out, np.round(sc.extents, 3), sum(len(g.faces) for g in sc.geometry.values()), "tris",
              round(os.path.getsize(out) / 1e6, 2), "MB")
