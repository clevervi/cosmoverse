import base64
import pathlib

image = pathlib.Path("docs/explorador_cmb.jpg")
payload = base64.b64encode(image.read_bytes()).decode("ascii")
out = pathlib.Path("docs/explorador_cmb.js")
out.write_text(
    "window.COSMOVerseCMB = {\n"
    '  source: "Planck CMB all-sky map, ESA/Planck (via NASA/IPAC)",\n'
    f'  data: "data:image/jpeg;base64,{payload}"\n'
    "};\n",
    encoding="utf-8",
    newline="\n",
)
print(out, out.stat().st_size)
