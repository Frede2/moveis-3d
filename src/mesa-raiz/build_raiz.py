"""Mesa de centro com base em raiz (modelo Meshy) + tampo de vidro real.
Entrada: dec.obj (raiz já cortada abaixo do vidro falso e reduzida) + mapas normal/rugosidade do GLB original.
Saída: mesa_raiz.glb (vidro com transmissão, web) e mesa_raiz_ar.glb (vidro simples, AR)."""
import io, json, struct, sys
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

YCUT, YMIN = 0.412, -0.4939259886741638   # cut height / floor in Meshy units
BASE_H = 0.438                              # root height in metres (glass 10 mm + pads -> 45 cm)
S = BASE_H / (YCUT - YMIN)
R_GLASS, T_GLASS, GAP = 0.45, 0.010, 0.002


def jpg(im, q):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); b.seek(0)
    out = Image.open(b); out.load(); out.format = "JPEG"; return out


# ---------- root
import os
root = trimesh.load(os.environ.get("DEC", "dec.obj"), process=False)
if isinstance(root, trimesh.Scene):
    root = list(root.geometry.values())[0]
fn = root.face_normals; fc = root.triangles_center
top_mask = (fn[:, 1] > 0.80) & (fc[:, 1] > YCUT - 0.09)
# keep only the biggest connected flat patch (the trunk's cut top); small patches on branches stay as bark
idx = np.where(top_mask)[0]
adj = root.face_adjacency
keep = np.isin(adj, idx).all(1)
comps = trimesh.graph.connected_components(adj[keep], nodes=idx, min_len=1)
areas = [root.area_faces[c].sum() for c in comps]
top_mask = np.zeros(len(root.faces), bool); top_mask[comps[int(np.argmax(areas))]] = True
trunk_top = root.submesh([np.where(top_mask)[0]], append=True)
root = root.submesh([np.where(~top_mask)[0]], append=True)
print("trunk-top faces:", top_mask.sum())
uv = root.visual.uv.copy()
orig = trimesh.load("in.glb")
om = list(orig.geometry.values())[0].visual.material
col = Image.open("material_0.png").convert("RGB")
nor = om.normalTexture.convert("RGB")
mr = om.metallicRoughnessTexture.convert("RGB").resize((1024, 1024), Image.LANCZOS)
ROOT = PBRMaterial(name="raiz_teca_branqueada", baseColorTexture=jpg(col, 86), normalTexture=jpg(nor, 90),
                   metallicRoughnessTexture=jpg(mr, 88), metallicFactor=0.0, roughnessFactor=1.0)

# ---------- caps on the cut tops (seen through the glass): end grain with rings
sec = root.section(plane_origin=[0, YCUT - 1e-4, 0], plane_normal=[0, 1, 0])
caps_v, caps_f, caps_uv = [], [], []
if sec is not None:
    planar, T = sec.to_planar()
    for poly in planar.polygons_full:
        if poly.area < 1e-5:
            continue
        v2, f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
        cx, cy = poly.centroid.x, poly.centroid.y
        rad = max(np.hypot(*(np.array(poly.exterior.coords) - [cx, cy]).T).max(), 1e-3)
        v3 = trimesh.transform_points(np.c_[v2, np.zeros(len(v2))], T)
        base = sum(len(a) for a in caps_v)
        caps_v.append(v3); caps_f.append(f + base)
        caps_uv.append(0.5 + (v2 - [cx, cy]) / (2.15 * rad))
if len(trunk_top.faces):
    tv = trunk_top.vertices[:, [0, 2]]
    c0 = tv.mean(0); rad = np.hypot(*(tv - c0).T).max()
    base = sum(len(a) for a in caps_v)
    caps_v.append(trunk_top.vertices.copy()); caps_f.append(trunk_top.faces + base)
    caps_uv.append(0.5 + (tv - c0) / (2.15 * rad))
caps = trimesh.Trimesh(np.vstack(caps_v), np.vstack(caps_f), process=False)
down = caps.face_normals[:, 1] < 0          # every cap faces up
caps.faces[down] = caps.faces[down][:, ::-1]
cap_uv = np.vstack(caps_uv)
print("caps:", len(caps_v), "polygons,", len(caps.faces), "tris")

# end-grain texture: honey rings, white-washed rim
N = 512
yy, xx = np.mgrid[0:N, 0:N] / N - 0.5
rng = np.random.default_rng(3)
warp = sum(rng.normal() * 0.012 * np.sin(2 * np.pi * (k * xx * rng.uniform(.5, 1.5) + k * yy * rng.uniform(.5, 1.5)) + rng.uniform(0, 6)) for k in (2, 3, 5))
r = np.hypot(xx, yy) + warp
ang = np.arctan2(yy, xx)
r2 = r * (1 + 0.06 * np.sin(3 * ang + 1.3) + 0.04 * np.sin(7 * ang))
freq = 34 + 6 * np.sin(r2 * 9)
rings = (0.5 + 0.5 * np.sin(r2 * 2 * np.pi * freq)) ** 4
fine = (0.5 + 0.5 * np.sin(r2 * 2 * np.pi * 120 + rng.normal(0, .3, (N, N)))) * 0.25
cracks = np.zeros((N, N))
for a0 in rng.uniform(-np.pi, np.pi, 5):
    d = np.abs(np.angle(np.exp(1j * (ang - a0))))
    cracks = np.maximum(cracks, np.clip(1 - d / 0.012, 0, 1) * (r > 0.08) * (r < rng.uniform(0.25, 0.45)))
honey = np.array([0.80, 0.72, 0.60]); dark = np.array([0.60, 0.50, 0.38]); white = np.array([0.92, 0.90, 0.86])
k = np.clip(0.55 * rings + fine, 0, 1)[..., None]
c = honey * (1 - k) + dark * k
c = c * (1 - 0.6 * cracks[..., None]) + np.array([0.35, 0.28, 0.2]) * 0.6 * cracks[..., None]
rim = np.clip((r - 0.40) / 0.07, 0, 1)[..., None]
c = c * (1 - rim) + white * rim
c *= (0.95 + 0.05 * rng.random((N, N)))[..., None]
ENDG = PBRMaterial(name="topo_cortado", baseColorTexture=Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)),
                   metallicFactor=0.0, roughnessFactor=0.8)


# ---------- glass disc (faces + polished edge as separate parts)
def glass(seg=160):
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    y0, y1 = BASE_H + GAP, BASE_H + GAP + T_GLASS
    V, F = [], []
    for y, up in ((y1, 1), (y0, -1)):
        b = len(V); V.append([0, y, 0])
        V += [[R_GLASS * cx, y, R_GLASS * cz] for cx, cz in ring]
        for i in range(seg):
            j = (i + 1) % seg
            F.append([b, b + 1 + j, b + 1 + i] if up > 0 else [b, b + 1 + i, b + 1 + j])
    faces = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    E, EF, EN = [], [], []
    for i in range(seg):
        for y in (y0, y1):
            E.append([R_GLASS * ring[i, 0], y, R_GLASS * ring[i, 1]]); EN.append([ring[i, 0], 0, ring[i, 1]])
    for i in range(seg):
        j = (i + 1) % seg
        a0, a1, b0, b1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        EF += [[a0, a1, b1], [a0, b1, b0]]
    edge = trimesh.Trimesh(np.array(E), np.array(EF), vertex_normals=np.array(EN), process=False)
    if (edge.face_normals * edge.vertex_normals[edge.faces].mean(1)).sum(1).mean() < 0:
        edge.invert()
    return faces, edge


g_faces, g_edge = glass()
GLASS = PBRMaterial(name="vidro", baseColorFactor=[0.04, 0.06, 0.055, 0.16], metallicFactor=0.0, roughnessFactor=0.0, alphaMode="BLEND")
EDGE = PBRMaterial(name="vidro_borda", baseColorFactor=[0.10, 0.20, 0.16, 0.95], metallicFactor=0.0, roughnessFactor=0.05, alphaMode="BLEND")
PAD = PBRMaterial(name="silicone", baseColorFactor=[0.6, 0.6, 0.6, 0.25], metallicFactor=0.0, roughnessFactor=0.3, alphaMode="BLEND")


def place(m):
    m = m.copy()
    m.apply_scale(S)
    m.apply_translation([0, -YMIN * S, 0])
    return m


def scene():
    sc = trimesh.Scene()
    r_ = place(root); r_.visual = TextureVisuals(uv=uv, material=ROOT)
    c_ = place(caps); c_.visual = TextureVisuals(uv=cap_uv, material=ENDG)
    sc.add_geometry(r_, node_name="raiz", geom_name="raiz")
    sc.add_geometry(c_, node_name="topos_cortados", geom_name="topos_cortados")
    for nm, m, mat in (("tampo_vidro", g_faces, GLASS), ("tampo_vidro_borda", g_edge, EDGE)):
        mm = m.copy(); mm.visual = TextureVisuals(material=mat); sc.add_geometry(mm, node_name=nm, geom_name=nm)
    return sc


def patch_glb(path, fn):
    data = open(path, "rb").read()
    jlen = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + jlen]); rest = data[20 + jlen:]
    fn(js)
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    open(path, "wb").write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + len(rest)) + struct.pack("<I4s", len(jb), b"JSON") + jb + rest)


def transmission(js):
    ext = js.setdefault("extensionsUsed", [])
    for e in ("KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_ior"):
        if e not in ext: ext.append(e)
    for m in js["materials"]:
        if m.get("name") == "vidro":
            m["pbrMetallicRoughness"]["baseColorFactor"] = [1, 1, 1, 1]; m["alphaMode"] = "OPAQUE"
            m["extensions"] = {"KHR_materials_transmission": {"transmissionFactor": 1.0},
                               "KHR_materials_volume": {"thicknessFactor": T_GLASS, "attenuationDistance": 0.25, "attenuationColor": [0.86, 0.95, 0.91]},
                               "KHR_materials_ior": {"ior": 1.52}}
        if m.get("name") == "vidro_borda":
            m["pbrMetallicRoughness"]["baseColorFactor"] = [0.35, 0.55, 0.47, 1]; m["alphaMode"] = "OPAQUE"
            m["extensions"] = {"KHR_materials_transmission": {"transmissionFactor": 0.5},
                               "KHR_materials_volume": {"thicknessFactor": 0.05, "attenuationDistance": 0.02, "attenuationColor": [0.22, 0.42, 0.34]},
                               "KHR_materials_ior": {"ior": 1.52}}


if __name__ == "__main__":
    sc = scene()
    print("extents (m):", np.round(sc.extents, 3), "tris:", sum(len(g.faces) for g in sc.geometry.values()))
    if os.environ.get("DEC"):                 # light version only for the iPhone (USDZ) export
        sc.export("mesa_raiz_ios.glb"); outs = ("mesa_raiz_ios.glb",)
    else:
        sc.export("mesa_raiz_ar.glb")
        sc.export("mesa_raiz.glb"); patch_glb("mesa_raiz.glb", transmission); outs = ("mesa_raiz.glb", "mesa_raiz_ar.glb")
    for f in outs:
        print(f, round(os.path.getsize(f) / 1e6, 2), "MB")
