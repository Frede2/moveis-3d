import sys, asyncio, os, shutil
from playwright.async_api import async_playwright
# usage: shoot.py model.glb out_prefix "orbit1|orbit2|..." [target] [fov]
glb, out = sys.argv[1], sys.argv[2]
orbits = sys.argv[3].split("|")
target = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else ""
fov = sys.argv[5] if len(sys.argv) > 5 else ""
shutil.copy(glb, "/home/claude/render/model.glb")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome",
            args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width":900,"height":700})
        shots=[]
        for i,o in enumerate(orbits):
            q=f"src=model.glb&orbit={o}"+(f"&target={target}" if target else "")+(f"&fov={fov}" if fov else "")
            await pg.goto(f"http://127.0.0.1:8765/view.html?{q}")
            try:
                await pg.wait_for_function("document.title==='ready'||document.title==='error'", timeout=90000)
            except Exception as e:
                print("timeout", e)
            print(o, await pg.title())
            f=f"{out}_{i}.png"; await pg.screenshot(path=f); shots.append(f)
        await b.close()
    from PIL import Image
    ims=[Image.open(f) for f in shots]
    W=sum(im.width for im in ims)//1; sheet=Image.new("RGB",(450*len(ims),350),"white")
    for i,im in enumerate(ims): sheet.paste(im.resize((450,350)),(i*450,0))
    sheet.save(out+"_sheet.png")
asyncio.run(main())
