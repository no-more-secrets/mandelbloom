# mandelgpu

Mandelbrot viewer for Windows. CUDA does the iteration, SDL3 and Dear ImGui
do the window and UI, OpenGL shows the result through CUDA interop (no copies).

## How it renders

1. The view centre is an MPFR number. Its precision follows the zoom.
2. The CPU iterates one reference orbit at that centre (`reference.cpp`).
3. The GPU iterates every pixel as a small delta from the reference
   (perturbation with rebasing and bilinear approximation) and writes a
   float field per pixel: smooth iteration count, distance estimate in
   pixels, exterior normal, final angle. The default kernel is float with a
   per-value exponent on every small quantity (`iterate_float.cuh`), so
   depth is not limited by float range; `--double` selects the double
   kernel (`render_cuda.cu`) for comparison.
4. A separate shading kernel colors the field each frame (`shade.cuh`).
   Palette changes and animation never re-iterate.

## Build

Requires Visual Studio 2026 with the C++ workload, CUDA Toolkit 13.4, and an
NVIDIA GPU. Dependencies come from the vcpkg bundled with Visual Studio.

```powershell
.\build.ps1 -Run
```

## Controls

| Input | Action |
|---|---|
| Drag | Pan |
| Wheel | Zoom around the cursor |
| R | Reset view |
| Tab | Hide or show the overlay |
| Esc | Quit |

## Shading

Shading never re-iterates. The Shading section of the overlay has presets
(classic, pastel lines, relief, mono lines) and the controls behind them:
cosine palette, slope lighting from the exterior normal, antialiased lines on
iteration band boundaries, distance-estimate edge darkening, palette
animation.

## Testing without touching the desktop

`--script "wait:1500;wheel:5;drag:300,120,12;pan:8,0;shot:out.png;quit"`
replays input through SDL events and saves the composited frame, with field
statistics on stdout. `--preset N` picks a shading preset, `--ss N`
supersampling, `--nobla` and `--double` select the slow paths for
comparison. Run it minimised.

## Plan

1. Double-precision kernel, interop display, pan and zoom. (done)
2. Perturbation: MPFR reference orbit on the CPU, deltas on the GPU. (done)
3. Bilinear approximation tables. (done)
4. Float kernel with exponents (floatexp). (done)
5. Progressive refinement, generational field, tweened zoom, pan inertia. (done)
6. Shaders: gradient/cosine palettes, slope lighting, iteration lines, modes, presets. (done)
7. Supersampling 1x/2x/3x. (done)
8. Better BLA validity (Imagina-style) for spiral regions.
9. HDR output (D3D12 swapchain), 4K export, zoom video export.
10. CPU SIMD path (ISPC) for glitch repair and second references.

Timings on an RTX 4080, 1920x1200, 1x:

| view | double | float |
|---|---|---|
| 8e5 zoom, 4000 iterations | 770 ms | 150 ms |
| 1.7e25 zoom, 20000 iterations | 406 ms | 88 ms |
