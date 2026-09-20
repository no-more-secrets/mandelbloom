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
find_package(glad CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(mandelgpu
  src/main.cpp
  src/gl_display.cpp
  src/render_cuda.cu
)
target_include_directories(mandelgpu PRIVATE src)
target_link_libraries(mandelgpu PRIVATE
  SDL3::SDL3
  imgui::imgui
  glad::glad
  CUDA::cudart
)
target_compile_definitions(mandelgpu PRIVATE SDL_MAIN_HANDLED)
if(MSVC)
  target_compile_options(mandelgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
endif()
target_compile_options(mandelgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
set_target_properties(mandelgpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
