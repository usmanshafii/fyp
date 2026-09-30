# KAL target model

Specification: section 2 of `FYP_Build_Spec_Multispectral_Aerial_Detection.md`.

```
pip install -r requirements.txt
python target/kal_geometry.py        # metre summary + closed-mesh check
python target/kal_measure.py         # mesh-vs-YAML invariants + hot-part visibility by aspect
pytest tests -q                      # 41 tests
python tools/photo_check.py          # overlays in docs/reference/, RMS per photo
python tools/draw_reference.py       # docs/reference/kal_views.png/.svg (top, side, front, rear)
python tools/make_decals.py          # assets/decals/ (already generated)
blender -b -P target/build_target.py -- --variants 5 --out out/phase1
```

Edit dimensions only in `configs/target.yaml`, then rerun everything above.
PyYAML is optional: without it (e.g. inside Blender) the bundled `target/_yamlite.py` parses the config.
