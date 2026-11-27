"""Poltrona giratória (modelo Meshy 7.1 a partir de 2 fotos) -> 78 x 90 x 99 cm, base cromada refeita.
Passos: corta a base gerada pela IA (cromado sai escuro/sujo), reduz o estofo com textura, faz base em trompete
cromada + coluna preta, junta 'sheen' (efeito veludo) ao tecido na versão web."""
import io, json, os, struct, subprocess, sys
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

H_REAL = 0.99
CUT_Y = -0.545                  # Meshy units: below this is the AI base / mechanism
STEP = sys.argv[1] if len(sys.argv) > 1 else "all"

src = trimesh.load("in.glb")
g = list(src.geometry.values())[0]
g.apply_transform(src.graph[src.graph.nodes_geometry[0]][0])
mat = g.visual.material
ymin = g.vertices[:, 1].min()
S = H_REAL / (g.vertices[:, 1].max() - ymin)
low = g.vertices[:, 1] < -0.85
CX, CZ = np.median(g.vertices[low][:, [0, 2]], axis=0)

if STEP in ("all", "cut"):
    up = g.slice_plane([0, CUT_Y, 0], [0, 1, 0])
    up.export("up.obj")
    print("upholstery faces:", len(up.faces))
    code = f"""
import pymeshlab
ms=pymeshlab.MeshSet(); ms.load_new_mesh('up.obj')
ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum={int(os.environ.get('FACES', 120000))}, preserveboundary=True, boundaryweight=2.0, qualitythr=0.4, extratcoordw=1.0, planarquadric=True)
ms.save_current_mesh('up_dec.obj'); print('decimated', ms.current_mesh().face_number())
"""
    subprocess.run([sys.executable, "-c", code], check=True, stderr=subprocess.DEVNULL)


def jpg(im, q):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); b.seek(0)
    out = Image.open(b); out.load(); out.format = "JPEG"; return out


up = trimesh.load("up_dec.obj", process=False)
if isinstance(up, trimesh.Scene):
    up = list(up.geometry.values())[0]
uv = up.visual.uv.copy()
col = Image.open([f for f in os.listdir(".") if f.startswith("material") and f.endswith(".png")][0]).convert("RGB")
FABRIC = PBRMaterial(name="tecido_chenille", baseColorTexture=jpg(col, 86), normalTexture=jpg(mat.normalTexture, 90),
                     metallicRoughnessTexture=jpg(mat.metallicRoughnessTexture.resize((1024, 1024), Image.LANCZOS), 88),
                     metallicFactor=0.0, roughnessFactor=1.0)
CHROME = PBRMaterial(name="cromado", baseColorFactor=[0.92, 0.92, 0.93, 1], metallicFactor=1.0, roughnessFactor=0.07)
BLACK = PBRMaterial(name="plastico_preto", baseColorFactor=[0.02, 0.02, 0.02, 1], metallicFactor=0.0, roughnessFactor=0.45)


def place(m):
    m = m.copy(); m.apply_translation([-CX, -ymin, -CZ]); m.apply_scale(S); return m


# ---- new base, built in metres around the origin (floor y=0)
R_DISC, T_DISC = 0.285, 0.012
H_TRUMP = 0.115                 # top of trumpet
seat_bottom = (CUT_Y - ymin) * S


def lathe(profile, sections=int(os.environ.get("LATHE", 160))):
    """profile: list of (r, y) from outside-bottom going in/up; revolved about Y."""
    pts = np.array(profile, float)
    m = trimesh.creation.revolve(pts, sections=sections)
    # revolve works in (x, z) -> rotate so the profile's 2nd coord is Y
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.merge_vertices()
    m.fix_normals()
    return m


t = np.linspace(0, 1, int(os.environ.get("PROF", 70)))
trump = [(R_DISC, 0.0), (R_DISC + 0.002, 0.004), (R_DISC, T_DISC)]
# concave trumpet from the disc to the stem
trump += [(0.032 + (R_DISC - 0.035) * (1 - tt) ** 2.6, T_DISC + (H_TRUMP - T_DISC) * tt) for tt in t[1:]]
trump += [(0.030, H_TRUMP + 0.004), (0.0, H_TRUMP + 0.004)]
base = lathe([(0.0, 0.0)] + trump)
collar = lathe([(0.0, H_TRUMP), (0.028, H_TRUMP), (0.028, H_TRUMP + 0.045), (0.0, H_TRUMP + 0.045)], 64)
# top of the column = lowest point of the seat shell near the axis (so the plate touches the fabric)
pu = place(up).vertices
near = np.hypot(pu[:, 0], pu[:, 2]) < 0.10
col_top = pu[near, 1].min() + 0.016 if near.any() else seat_bottom + 0.004
column = lathe([(0.0, H_TRUMP + 0.045), (0.022, H_TRUMP + 0.045)] +
               [(0.022 + 0.0025 * (k % 2), H_TRUMP + 0.045 + (col_top - H_TRUMP - 0.045) * k / 24) for k in range(1, 24)] +
               [(0.022, col_top), (0.0, col_top)], 48)
plate = trimesh.creation.box([0.15, 0.010, 0.12]); plate.apply_translation([0, col_top - 0.005, 0])
knob = trimesh.creation.cylinder(radius=0.014, height=0.05, sections=24); knob.apply_translation([0.05, col_top - 0.03, 0.02])

parts = [("estofo", place(up), FABRIC, uv), ("base_trompete", base, CHROME, None), ("anel", collar, CHROME, None),
         ("coluna", column, BLACK, None), ("placa", plate, BLACK, None), ("manipulo", knob, BLACK, None)]


def export(path):
    sc = trimesh.Scene()
    for name, m, material, u in parts:
        mm = m.copy()
        mm.visual = TextureVisuals(uv=u, material=material) if u is not None else TextureVisuals(material=material)
        sc.add_geometry(mm, node_name=name, geom_name=name)
    sc.export(path)
    print(path, "extents (m):", np.round(sc.extents, 3), "tris:", sum(len(x.faces) for x in sc.geometry.values()),
          round(os.path.getsize(path) / 1e6, 2), "MB")


def patch(path, fn):
    data = open(path, "rb").read(); jlen = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + jlen]); rest = data[20 + jlen:]; fn(js)
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    open(path, "wb").write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + len(rest)) + struct.pack("<I4s", len(jb), b"JSON") + jb + rest)


def sheen(js):
    js.setdefault("extensionsUsed", []).append("KHR_materials_sheen")
    for m in js["materials"]:
        if m.get("name") == "tecido_chenille":
            m["extensions"] = {"KHR_materials_sheen": {"sheenColorFactor": [0.55, 0.53, 0.48], "sheenRoughnessFactor": 0.55}}


out = os.environ.get("OUT", "poltrona")
export(out + ".glb")
if out == "poltrona":
    patch(out + ".glb", sheen)
