#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "image_io.h"

bool writePng(const char* path, const uint32_t* rgba, int width, int height) {
    return stbi_write_png(path, width, height, 4, rgba, width * 4) != 0;
}

#include <tinyexr.h>  // compiled library from vcpkg; no IMPLEMENTATION define here

bool writeExr(const char* path, const float* rgb, int width, int height) {
    const char* err = nullptr;
    const int ret = SaveEXR(rgb, width, height, 3, 1, path, &err);
    if (err) FreeEXRErrorMessage(err);
    return ret == TINYEXR_SUCCESS;
}
