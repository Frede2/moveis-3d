"""Sofá de canto bouclé (Meshy 7.1, 3 fotos) -> 267 x 165 x 73 cm, tecido bouclé baked (boucle.py)."""
import io, json, os, struct, sys
import numpy as np, trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

def jpg(arr_or_path, q, size=None):
    im = Image.open(arr_or_path).convert("RGB") if isinstance(arr_or_path, str) else arr_or_path
    if size: im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); o = Image.open(b); o.load(); o.format = "JPEG"; return o

col = np.asarray(Image.open("bq_col.png")).astype(np.float32) / 255
lum = (col * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
col = np.clip((col * 0.55 + lum * 0.45) * [1.0, 0.985, 0.95], 0, 1)          # greyer warm beige, like the photo
M = PBRMaterial(name="tecido_boucle", baseColorTexture=jpg(Image.fromarray((col * 255).astype(np.uint8)), 88),
                normalTexture=jpg("bq_nor.png", 92), metallicRoughnessTexture=jpg("bq_mr.png", 88, (1024, 1024)),
                metallicFactor=0.0, roughnessFactor=1.0)

def build(obj, out, sheen):
    m = trimesh.load(obj, process=False)
    if isinstance(m, trimesh.Scene): m = list(m.geometry.values())[0]
    uv = m.visual.uv; m.apply_scale([1, 1, 1.65 / 1.709])   # match the 165 cm depth of the spec
    m.visual = TextureVisuals(uv=uv, material=M)
    sc = trimesh.Scene(); sc.add_geometry(m, node_name="sofa", geom_name="sofa"); sc.export(out)
    if sheen:
        data = open(out, "rb").read(); jl = struct.unpack("<I", data[12:16])[0]; js = json.loads(data[20:20 + jl]); rest = data[20 + jl:]
        js.setdefault("extensionsUsed", []).append("KHR_materials_sheen")
        js["materials"][0]["extensions"] = {"KHR_materials_sheen": {"sheenColorFactor": [0.5, 0.48, 0.44], "sheenRoughnessFactor": 0.6}}
        jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
        open(out, "wb").write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + len(rest)) + struct.pack("<I4s", len(jb), b"JSON") + jb + rest)
    print(out, np.round(sc.extents, 3), len(m.faces), round(os.path.getsize(out) / 1e6, 2), "MB")

build("dec120.obj", "sofa.glb", True)
build("dec120.obj", "sofa_ar.glb", False)
build("dec60.obj", "sofa_ios.glb", False)
