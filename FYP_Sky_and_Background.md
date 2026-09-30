# Sky and Background — Rendering Setup

Companion to `FYP_Build_Spec_Multispectral_Aerial_Detection.md` (section 4 step 7). Moved here unchanged so the main spec can focus on the drone model; not yet re-reviewed.

## 1. Sky and background

An equirectangular HDRI spreads its width over 360°, so the pixels available across a frame are `hdri_width × HFOV / 360`. At 10° HFOV a 16K HDRI yields about 455 real pixels stretched across a 1280-pixel supersampled render — visibly blurred and stair-stepped. A blurred sky makes 2–4 px targets artificially easy to find, biasing exactly the result this project measures.

| Camera | Render width | HDRI width needed | Verdict |
| --- | --- | --- | --- |
| 50° HFOV | 1280 (2×) | 9,216 | 16K works |
| 50° HFOV | 2560 (4×) | 18,432 | 16K marginal |
| 10° HFOV | 1280 (2×) | 46,080 | no HDRI is large enough |

Use the procedural **Sky Texture (Nishita; "Single Scattering" in Blender 5.x)**, which is resolution-independent and physically parameterised. CC0 HDRIs stay available as an optional variety source for the 50° camera only, at 16K.

| Setting | Value |
| --- | --- |
| Sun Disc | **Off** — Cycles samples a small bright disc in the world poorly; use a Sun lamp instead |
| Sun Elevation / Rotation | 15–60° day, 0–360°; the Sun lamp is driven from the same YAML values |
| Air / Dust / Ozone | 1.0–2.0 / 0.5–6.0 / 1.0–3.0; the high dust end matches arid scenes |
| Sun lamp | Angle 0.526° (true solar disc), strength 700–1000 W/m² at high elevation, scaled roughly with sin(elevation) |
| Focal length | 38.6 mm (50° HFOV) or 205.7 mm (10°) on a 36 mm sensor, sensor fit horizontal |
| Camera clip | Start 0.1 m, **End 20,000 m** — the 100 m default silently clips every target |
| Film → Filter Size | 0.01 px — the 1.5 px default double-blurs against the PSF in the main spec, section 4 step 1 |
| Colour management | View Transform Standard, Look None, Exposure 0 |
| Light paths | Total 4, diffuse 2, glossy 2, transmission 2, volume 0 |
| Samples | 256, adaptive threshold 0.01, denoising off |
| Persistent Data | On, for many frames from one scene |

**Haze comes from the Mist pass, not volumetrics.** Kilometre-scale volumetric atmosphere in Cycles is prohibitively slow. Enable View Layer → Passes → Data → Mist (start 0, depth 20,000 m) and apply extinction in the loader, so visibility can be swept without re-rendering:

```latex
I_{\text{out}} = I_{\text{scene}}\,e^{-k d} + L_{\text{horizon}}\left(1 - e^{-k d}\right), \qquad k = \frac{3.912}{V}
```

V is meteorological visibility in metres. At V = 10 km a target at 3.7 km retains only 24% of its contrast, which alone dominates results in the smallest bins.

**Ground plane (horizon family):** 20 km × 20 km, lightly subdivided and noise-displaced for low hills; diffuse albedo 0.25–0.35, roughness 0.9; its own material slot so LWIR and SWIR can assign a temperature offset and reflectance. Camera height 1.5–3 m; look elevation +10° to +60° for open sky, −1° to +5° for horizon scenes.

**Acceptance test before the LWIR pass.** Render one target-free frame at 10° and confirm: a real sky gradient (bright near horizon and sun, darker at zenith), no pixelation or stair-stepping, and — most important — measure a flat sky patch and check that its standard deviation over mean is under about 0.5%. If render noise is comparable to the contrast a 2–4 px target produces, the experiment measures Cycles sampling noise rather than detectability.

## 2. Phase 2 brief additions (sky, lighting, ground, haze)

```
- sky: procedural Sky Texture (Nishita), NOT an HDRI (an HDRI cannot resolve a 10 deg field of view); sun disc off, with a separate Sun lamp at angle 0.526 deg and strength 700-1000 W/m2 driven from the same sun elevation/rotation
- lighting: day and dusk (sun elevation -6 to 60 deg, including sun near the line of sight); air 1.0-2.0, dust 0.5-6.0, ozone 1.0-3.0; night at low illumination
- ground plane for horizon scenes: 20 km square, noise-displaced, albedo 0.25-0.35, own material slot
- enable the Mist pass (start 0, depth 20000 m) so haze can be applied in the loader instead of volumetrically
```
