# mandelgpu

Mandelbrot viewer for Windows. CUDA does the iteration, SDL3 and Dear ImGui
do the window and UI, OpenGL shows the result through CUDA interop (no copies).

## How it renders

1. The view centre is an MPFR number. Its precision follows the zoom.
2. The CPU iterates one reference orbit at that centre (`reference.cpp`).
3. The GPU iterates every pixel as a small delta from the reference
   (perturbation with rebasing) and writes a float field per pixel: smooth
   iteration count, distance estimate, final angle (`render_cuda.cu`).
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

## Plan

1. Double-precision kernel, interop display, pan and zoom. (done)
2. Perturbation: MPFR reference orbit on the CPU, deltas on the GPU. (done)
3. Bilinear approximation tables, floatexp for depths past 1e-300.
4. CPU SIMD path (ISPC) for tiles the GPU is not touching and for glitch repair.
5. Progressive refinement, coloring controls, image export.
