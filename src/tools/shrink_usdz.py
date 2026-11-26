"""Shrink a USDZ from three.js: PNG textures -> JPEG, keep 64-byte alignment (required by USDZ).
Usage: python3 shrink_usdz.py in.usdz out.usdz [quality]"""
import io
import sys
import zipfile
from PIL import Image


def shrink(src, dst, q=85):
    zin = zipfile.ZipFile(src)
    files = {}
    renames = {}
    for info in zin.infolist():
        data = zin.read(info.filename)
        name = info.filename
        if name.startswith("textures/") and name.endswith(".png"):
            im = Image.open(io.BytesIO(data))
            if im.mode in ("RGBA", "LA") and min(im.getchannel("A").getextrema()) < 255:
                files[name] = data            # real transparency: keep PNG
                continue
            b = io.BytesIO()
            im.convert("RGB").save(b, "JPEG", quality=q, subsampling=0)
            new = name[:-4] + ".jpg"
            renames[name] = new
            name, data = new, b.getvalue()
        files[name] = data
    for name in list(files):
        if name.endswith(".usda"):
            txt = files[name].decode("utf-8")
            for old, new in renames.items():
                txt = txt.replace(old, new)
            files[name] = txt.encode("utf-8")
    with open(dst, "wb") as fp:
        z = zipfile.ZipFile(fp, "w", zipfile.ZIP_STORED)
        for name, data in files.items():   # model.usda stays first
            zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            off = fp.tell() + 30 + len(name.encode())
            pad = (64 - (off + 4) % 64) % 64
            zi.extra = b"\x86\x19" + pad.to_bytes(2, "little") + b"\0" * pad
            z.writestr(zi, data)
        z.close()
    # check alignment
    for i in zipfile.ZipFile(dst).infolist():
        start = i.header_offset + 30 + len(i.filename.encode()) + len(i.extra)
        assert start % 64 == 0, i.filename


if __name__ == "__main__":
    shrink(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 85)
