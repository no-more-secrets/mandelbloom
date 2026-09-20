cmake_minimum_required(VERSION 3.28)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)   # Ada (RTX 40xx); must precede project()
endif()
project(mandelgpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(SDL3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Stb REQUIRED)
find_path(MPFR_INCLUDE_DIR mpfr.h REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

add_executable(mandelgpu
  src/main.cpp
  src/dx_display.cpp
  src/reference.cpp
  src/bla.cpp
  src/script.cpp
  src/image_io.cpp
  src/render_cuda.cu
)
target_include_directories(mandelgpu PRIVATE src ${MPFR_INCLUDE_DIR} ${Stb_INCLUDE_DIR})
target_link_libraries(mandelgpu PRIVATE
  SDL3::SDL3
  imgui::imgui
  d3d12
  dxgi
  dxguid
  d3dcompiler
  CUDA::cudart
  ${MPFR_LIBRARY}
  ${GMP_LIBRARY}
)
target_compile_definitions(mandelgpu PRIVATE SDL_MAIN_HANDLED NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(mandelgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
endif()
target_compile_options(mandelgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
set(MANDELGPU_OUTPUT_NAME "mandelgpu" CACHE STRING "Executable name (lets a test build coexist with a running one)")
set_target_properties(mandelgpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                           OUTPUT_NAME "${MANDELGPU_OUTPUT_NAME}")
