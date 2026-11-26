import sys, asyncio, base64
from playwright.async_api import async_playwright
async def main(names):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
        pg = await b.new_page()
        for n in names:
            await pg.goto(f"http://127.0.0.1:8766/export_any.html?src={n}.glb")
            await pg.wait_for_function("document.title==='done'||document.title==='error'", timeout=180000)
            if await pg.title() == "error":
                print(n, "ERR", await pg.evaluate("window.ERR")); continue
            data = base64.b64decode(await pg.evaluate("window.USDZ"))
            open(f"{n}.usdz","wb").write(data); print(n, round(len(data)/1e6,2), "MB")
        await b.close()
asyncio.run(main(sys.argv[1:]))
