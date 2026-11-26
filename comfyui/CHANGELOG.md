# Registry release notes

## 0.1.2

Fix installation through ComfyUI-Manager and `comfy node install nobg`: the
pack now ships a `requirements.txt` declaring `nobg>=0.3.2`, so the `nobg`
library is installed alongside the node. 0.1.1 installed the node without the
library and failed to load with `No module named 'nobg'`. No node changes.

## 0.1.1

Add the NoBG logo as the Registry node icon. No node changes.

## 0.1.0

Initial NoBG ComfyUI Registry release: model loader, background removal with
optional SAM3 text prompt and foreground refinement, and a separate foreground
refinement node. The entrypoint delegates to the separately installed `nobg`
PyPI package. Legacy Python callers can pass boxes to `remove(...)`, but boxes
are not exposed as a ComfyUI input.
