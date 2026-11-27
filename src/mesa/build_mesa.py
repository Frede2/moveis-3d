"""Mesa de centro, tampo de vidro fumado 8 mm, 120 x 70 x 38 cm. Base original em aço pintado a preto:
dois quadros trapezoidais (mais largos no chão) ligados por uma barra longitudinal ao nível do chão.
Exporta mesa_vidro.glb (vidro com transmissão KHR) e mesa_vidro_ar.glb (vidro com transparência simples, para USDZ)."""
import json
import struct
import sys
import numpy as np
import trimesh
import fast_simplification
from skimage.measure import marching_cubes
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

W, D, H = 1.20, 0.70, 0.38
GT = 0.008                     # glass thickness
GLASS_Y = H - GT - 0.003       # underside of glass (3 mm felt glides below the frames)
PAD = 0.003                    # clear silicone pads between frame and glass
FY = GLASS_Y - PAD             # top of frames
BAR_W, BAR_T = 0.040, 0.016    # flat bar: in-plane width, thickness
FX = 0.42                      # frames at x = +-FX
TOP_HALF, BOT_HALF = 0.235, 0.305   # half lengths (z) of top and floor bars, outer edge
R_EDGE = 0.003


def lin(*c):
    return [(x / 255) ** 2.2 for x in c] + [1.0]


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


def sd_convex(py, pz, poly):
    """Signed distance in the (y,z) plane to a convex polygon given counter-clockwise."""
    d = np.full(py.shape, np.inf)
    inside = np.ones(py.shape, bool)
    n = len(poly)
    for i in range(n):
        (ay, az), (by, bz) = poly[i], poly[(i + 1) % n]
        ey, ez = by - ay, bz - az
        wy, wz = py - ay, pz - az
        t = np.clip((wy * ey + wz * ez) / (ey * ey + ez * ez), 0, 1)
        d = np.minimum(d, np.hypot(wy - ey * t, wz - ez * t))
        inside &= (ey * wz - ez * wy) >= 0      # polygons are counter-clockwise in (y,z)
    return np.where(inside, -d, d)


def trapezoid(inset):
    """Frame outline in (y,z): floor at y=0 (wide), top at y=FY (narrow), shrunk by `inset`."""
    ybot, ytop = 0.0 + inset, FY - inset
    # slanted sides: interpolate half-length with height, then move inward by inset (approx. along z)
    slope = (BOT_HALF - TOP_HALF) / FY
    zb = BOT_HALF - slope * ybot - inset * np.sqrt(1 + slope ** 2)
    zt = BOT_HALF - slope * ytop - inset * np.sqrt(1 + slope ** 2)
    return [(ybot, -zb), (ytop, -zt), (ytop, zt), (ybot, zb)]


OUT, INN = trapezoid(0.0), trapezoid(BAR_W)


def make_frame_sdf():
    def sd2(py, pz):
        a = sd_convex(py, pz, OUT)
        b = sd_convex(py, pz, INN)
        return np.maximum(a, -b)

    # verify orientation once
    cy = np.array([FY / 2]); cz = np.array([0.0])
    assert sd_convex(cy, cz, OUT)[0] < 0, "polygon orientation"

    def sdf(x, y, z):
        d2 = sd2(y, z) + R_EDGE
        dx = np.abs(x) - BAR_T / 2 + R_EDGE
        out = np.hypot(np.maximum(d2, 0), np.maximum(dx, 0))
        return out + np.minimum(np.maximum(d2, dx), 0) - R_EDGE
    return sdf


def sd_round_box(x, y, z, bx, by, bz, r):
    qx, qy, qz = np.abs(x) - bx + r, np.abs(y) - by + r, np.abs(z) - bz + r
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def rbox_mesh(ext, r, h, target):
    bx, by, bz = np.asarray(ext) / 2
    return mesh_from_sdf(lambda x, y, z: sd_round_box(x, y, z, bx, by, bz, r),
                         (-bx - 2 * h, -by - 2 * h, -bz - 2 * h), (bx + 2 * h, by + 2 * h, bz + 2 * h), h, target)


def glass_top():
    """Exact box with a small rounded (polished) edge, built analytically for clean flat faces."""
    b = np.array([W, GT, D]) / 2
    r = 0.0025
    t = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    coords = [np.unique(np.r_[-bi + r * t, bi - r * t[::-1]]) for bi in b]
    inner = b - r
    V, Fc, N = [], [], []
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
                    dd = p - q
                    n = dd / np.linalg.norm(dd)
                    V.append(q + n * r); N.append(n)
            n1, n2 = len(c1), len(c2)
            for i in range(n1 - 1):
                for j in range(n2 - 1):
                    a_, b_, c_, d_ = base + i * n2 + j, base + (i + 1) * n2 + j, base + (i + 1) * n2 + j + 1, base + i * n2 + j + 1
                    Fc += [[a_, b_, c_], [a_, c_, d_]]
    V, N, Fc = np.array(V), np.array(N), np.array(Fc)
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    flip = (fn * N[Fc].mean(1)).sum(1) < 0
    Fc[flip] = Fc[flip][:, ::-1]
    m = trimesh.Trimesh(V, Fc, vertex_normals=N, process=False)
    m.apply_translation([0, GLASS_Y + GT / 2, 0])
    flat = np.abs(N[Fc].mean(1)[:, 1]) > 0.98          # top / bottom faces
    faces = m.submesh([np.where(flat)[0]], append=True)
    edge = m.submesh([np.where(~flat)[0]], append=True)
    return faces, edge


# ---------------------------------------------------------------- geometry
from shapely.geometry import Polygon


def extrude_ring(outer, inner, thick):
    """Flat-shaded extrusion of a polygon ring drawn in (z, y), along x. Small chamfer-free, crisp powder-coated look."""
    poly = Polygon([(z, y) for y, z in outer], [[(z, y) for y, z in inner]])
    m = trimesh.creation.extrude_polygon(poly, thick)          # extrudes along +Z of the 2D (x=z, y=y) plane
    # map: 2D x -> world z, 2D y -> world y, extrusion -> world x
    T = np.array([[0, 0, 1, -thick / 2], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    m.unmerge_vertices()                                        # keep hard edges (flat normals)
    return m


frame = extrude_ring(OUT, INN, BAR_T)
print("frame bounds", np.round(frame.bounds, 4))

parts = []
for s in (-1, 1):
    f = frame.copy(); f.apply_translation([s * FX, 0, 0])
    parts.append((f"quadro_{'esq' if s < 0 else 'dir'}", f, "steel"))
# longitudinal floor bar between the frames (lies flat, 40 mm wide, 16 mm tall)
bar = trimesh.creation.box([2 * FX - BAR_T, BAR_T, BAR_W]); bar.unmerge_vertices()
bar.apply_translation([0, BAR_T / 2, 0])
parts.append(("barra_longitudinal", bar, "steel"))
# felt glides under the frames
glide = trimesh.creation.box([BAR_T - 0.002, 0.003, 0.03])
for s in (-1, 1):
    for sz in (-1, 1):
        g = glide.copy(); g.apply_translation([s * FX, -0.0015, sz * (BOT_HALF - 0.03)])
        parts.append((f"deslizador_{s}_{sz}", g, "felt"))
# clear silicone pads on top of the frames
pad = trimesh.creation.cylinder(radius=0.007, height=PAD, sections=24)
pad.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
slope = (BOT_HALF - TOP_HALF) / FY
for s in (-1, 1):
    for sz in (-1, 1):
        p = pad.copy(); p.apply_translation([s * FX, FY + PAD / 2, sz * (TOP_HALF - BAR_W / 2)])
        parts.append((f"batente_{s}_{sz}", p, "pad"))
g_faces, g_edge = glass_top()
parts.append(("tampo_vidro", g_faces, "glass"))
parts.append(("tampo_vidro_borda", g_edge, "glass_edge"))

STEEL = PBRMaterial(name="aco_pintado_preto", baseColorFactor=lin(26, 26, 27), metallicFactor=0.0, roughnessFactor=0.45)
FELT = PBRMaterial(name="feltro", baseColorFactor=lin(40, 40, 40), metallicFactor=0.0, roughnessFactor=0.95)
PADM = PBRMaterial(name="silicone", baseColorFactor=[0.6, 0.6, 0.6, 0.25], metallicFactor=0.0, roughnessFactor=0.3, alphaMode="BLEND")
# AR fallback: dark tint + low alpha reads as smoked glass (a light grey reads as milky plastic)
GLASS = PBRMaterial(name="vidro_fumado", baseColorFactor=[0.015, 0.018, 0.018, 0.50], metallicFactor=0.0, roughnessFactor=0.0,
                    alphaMode="BLEND", doubleSided=False)
# polished 8 mm edge: looks dark green-grey and almost opaque, the main cue that it is real glass
EDGE = PBRMaterial(name="vidro_borda", baseColorFactor=[0.045, 0.075, 0.068, 0.93], metallicFactor=0.0, roughnessFactor=0.05,
                   alphaMode="BLEND")
MATS = {"steel": STEEL, "felt": FELT, "pad": PADM, "glass": GLASS, "glass_edge": EDGE}


def export(path):
    scene = trimesh.Scene()
    for name, m, role in parts:
        mm = m.copy()
        mm.visual = TextureVisuals(material=MATS[role])
        scene.add_geometry(mm, node_name=name, geom_name=name)
    scene.apply_translation([0, -scene.bounds[0][1], 0])
    scene.export(path)
    print(path, "extents", np.round(scene.extents, 3), "tris", sum(len(g.faces) for g in scene.geometry.values()))


def patch_glb(path, fn):
    data = open(path, "rb").read()
    jlen = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:]
    fn(js)
    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + len(rest)) + struct.pack("<I4s", len(jb), b"JSON") + jb + rest
    open(path, "wb").write(out)


def add_transmission(js):
    """Web version: real glass (refraction + tinted volume). alpha kept as fallback for viewers without the extension."""
    ext = js.setdefault("extensionsUsed", [])
    for e in ("KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_ior"):
        if e not in ext:
            ext.append(e)
    for m in js["materials"]:
        if m.get("name") == "vidro_fumado":
            m["pbrMetallicRoughness"]["baseColorFactor"] = [0.92, 0.93, 0.93, 1.0]
            m["alphaMode"] = "OPAQUE"
            m["extensions"] = {
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_volume": {"thicknessFactor": GT, "attenuationDistance": 0.007,
                                          "attenuationColor": [0.30, 0.31, 0.31]},
                "KHR_materials_ior": {"ior": 1.52},
            }
        if m.get("name") == "vidro_borda":
            m["pbrMetallicRoughness"]["baseColorFactor"] = [0.30, 0.42, 0.39, 1.0]
            m["alphaMode"] = "OPAQUE"
            m["extensions"] = {
                "KHR_materials_transmission": {"transmissionFactor": 0.6},
                "KHR_materials_volume": {"thicknessFactor": 0.05, "attenuationDistance": 0.02,
                                          "attenuationColor": [0.22, 0.32, 0.29]},
                "KHR_materials_ior": {"ior": 1.52},
            }


if __name__ == "__main__":
    export("mesa_vidro_ar.glb")             # simple transparency: Android Scene Viewer / iPhone
    export("mesa_vidro.glb")
    patch_glb("mesa_vidro.glb", add_transmission)
