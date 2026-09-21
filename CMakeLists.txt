cmake_minimum_required(VERSION 3.28)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Turing (RTX 20xx), Ampere (A100, RTX 30xx), Ada (RTX 40xx), Blackwell
  # (RTX 50xx) as native code, plus PTX for whatever comes next. Must precede
  # project().
  set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 120-real 120-virtual)
endif()
project(mandelgpu VERSION 0.1.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(SDL3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyexr CONFIG REQUIRED)
find_path(MPFR_INCLUDE_DIR mpfr.h REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

add_executable(mandelgpu
  src/main.cpp
  src/dx_display.cpp
  src/presets.cpp
  src/reference.cpp
  src/bla.cpp
  src/script.cpp
  src/image_io.cpp
  src/video_encoder.cpp
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
  unofficial::tinyexr::tinyexr
  ${GMP_LIBRARY}
)
target_compile_definitions(mandelgpu PRIVATE SDL_MAIN_HANDLED NOMINMAX WIN32_LEAN_AND_MEAN
                                             MANDELGPU_VERSION="${PROJECT_VERSION}")
if(MSVC)
  target_compile_options(mandelgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
endif()
target_compile_options(mandelgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
set(MANDELGPU_OUTPUT_NAME "mandelgpu" CACHE STRING "Executable name (lets a test build coexist with a running one)")
set_target_properties(mandelgpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                           OUTPUT_NAME "${MANDELGPU_OUTPUT_NAME}")
# Icon and version resource.
if(WIN32)
  set(MANDELGPU_ICON "${CMAKE_SOURCE_DIR}/src/icon.ico")
  configure_file(src/app.rc.in "${CMAKE_BINARY_DIR}/app.rc" @ONLY)
  target_sources(mandelgpu PRIVATE "${CMAKE_BINARY_DIR}/app.rc")
  set_source_files_properties("${CMAKE_BINARY_DIR}/app.rc" PROPERTIES OBJECT_DEPENDS "${MANDELGPU_ICON}")
endif()
# Windowed app: no console window on launch. Keep the plain main() entry
# point; stdout still works when the launcher redirects it.
if(MSVC)
  target_link_options(mandelgpu PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()
