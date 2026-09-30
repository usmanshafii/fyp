import blenderproc as bproc  # noqa: E402,F401  must be the first code line for `blenderproc run`
__doc__ = """Phase 1 brief step 4: wrap build_target() for BlenderProc and render it with the Phase 2 camera
at 2x and 4x supersampling.

    blenderproc run tools/bproc_check.py --custom-blender-path "<Blender 4.2 folder>"

BlenderProc installs its pip packages into <blender folder>/custom-python-packages, so the Blender
folder must be writable (a copy outside Program Files, or BlenderProc's own download via
--blender-install-path). Writes out/bproc_check/{2x,4x}.png and prints the MeshObject summary.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import build_target as bt  # noqa: E402
from data.imgio import write_png  # noqa: E402
from data.writer import tonemap  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import placement as pl  # noqa: E402
from render import psf, rgb, sampling  # noqa: E402

bproc.init()
cfgs = sampling.load_configs()
rc = cfgs["render"]["render"]
out = os.path.join(ROOT, "out", "bproc_check")
os.makedirs(out, exist_ok=True)
obj = bt.build_target(None, name="KAL")
mesh_objs = bproc.object.convert_to_meshes([obj])
print("BlenderProc MeshObject:", mesh_objs[0].get_name(), "vertices:", len(mesh_objs[0].get_mesh().vertices),
      "material slots:", len(mesh_objs[0].get_materials()))
atmo = dict(air=1.0, dust=1.0, ozone=1.0)
calib = dict(sun=rgb.calibrate_sun(35.0, atmo, 128))       # note: resets data-blocks, rebuild below
bu.reset()
bu.setup_cycles(rc)
obj = bt.build_target(None, name="KAL")
bproc.object.convert_to_meshes([obj])
sc = bpy.context.scene
spec = dict(lighting="day", atmosphere=atmo, sun=dict(elevation_deg=35.0, azimuth_deg=120.0), clouds=None, airglow_rgb=0.0)
sc.world = rgb.world(spec, calib)
rgb.lamps(spec, calib)
W, H, hfov = 640, 512, 10.0
cam = pl.Camera(W, H, hfov, (0, 0, 2.0), 0.0, 12.0)
bproc.camera.set_intrinsics_from_blender_params(lens=pl.focal_mm(36.0, hfov), lens_unit="MILLIMETERS",
                                                image_width=W, image_height=H, clip_start=rc["clip_m"][0],
                                                clip_end=rc["clip_m"][1])
bpy.context.scene.camera.data.sensor_width = 36.0
bpy.context.scene.camera.data.sensor_fit = "HORIZONTAL"
bproc.camera.add_camera_pose(pl.matrix_world(cam.R, cam.C))
# a 24 px target (8-16 px bins would test 4x): place it along the optical axis
Vt = np.array([v.co[:] for v in obj.data.vertices])
d = cam.ray(W / 2, H / 2)
dist, box = pl.solve_distance(cam, Vt, np.eye(3), d, 24.0)
bu.set_matrix(obj, pl.matrix_world(pl.object_rotation(60.0), cam.C + d * dist))
for s in (2, 4):
    p = bu.render(os.path.join(out, f"{s}x.exr"), (W * s, H * s))
    dd = bu.read_passes(p, grey=False)
    img = psf.psf_integrate(dd["combined"] + dd["env"], cfgs["sensors"]["psf_sigma_px"]["rgb"][0], s)
    write_png(os.path.join(out, f"{s}x.png"), tonemap(img, "rgb"))
    print(f"{s}x supersampling: rendered {W * s}x{H * s}, target box {box}")
print("bproc check done:", out)
