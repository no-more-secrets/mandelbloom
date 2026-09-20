# mandelgpu

Mandelbrot viewer for Windows. CUDA does the iteration, SDL3 and Dear ImGui
do the window and UI, OpenGL shows the result through CUDA interop (no copies).

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

## Plan

1. Double-precision kernel, interop display, pan and zoom. (done)
2. Perturbation: arbitrary-precision reference orbit on the CPU, float deltas on the GPU.
3. Bilinear approximation tables, floatexp for depths past 1e-300.
4. CPU SIMD path (ISPC) for tiles the GPU is not touching and for glitch repair.
5. Progressive refinement, coloring controls, image export.
