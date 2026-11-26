# Código dos modelos

Tudo em Python (numpy, trimesh, scikit-image, fast-simplification, Pillow, OpenEXR, scipy) + Chrome headless (playwright).

## Cómoda v2 (atual) — `comoda/`
1. Texturas: descarregar da Poly Haven (CC0) `oak_wood_planks` e `japanese_cedar_planks` em 4K (formato "blend" zip, traz diff jpg, nor_gl exr, rough exr, disp png) e extrair para `/home/claude/tex_in/<nome>_4k_blend/`.
2. `python3 prep_tex.py` → tira as juntas do soalho (deteta-as no mapa de deslocamento) e grava texturas "tábua colada" em `/home/claude/comoda2/tex/`.
3. `python3 build_comoda2.py [natural fumado nordico cerejeira salvia]` → `comoda_<acabamento>.glb`.
   - `finishes.py` define os 5 acabamentos (graduação de cor, rugosidade, força do normal, puxador).
   - A flor das laterais é uma imagem PNG (caminho `FLOWER` no script), composta por `wood.image_layer`. Licença por confirmar.
4. USDZ: `tools/run_export.py comoda_natural ...` com `export_any.html` servido em http://127.0.0.1:8766 e three.js 0.169 em `node_modules` (`npm i three@0.169`).
5. Verificar: `tools/shoot.py modelo.glb prefixo "30deg 75deg 2.2m|..."` com `view.html` + `model-viewer.min.js` servidos em http://127.0.0.1:8765.

`comoda/build_comoda.py` é a v1 (texturas procedurais) — só referência.

## Cadeira — `cadeira/`
`build_chair2.py` (usa `textures.py`) → `cadeira_detalhada.glb`. Ainda com texturas procedurais: é a próxima peça a melhorar.
