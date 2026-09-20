#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "image_io.h"

bool writePng(const char* path, const uint32_t* rgba, int width, int height) {
    return stbi_write_png(path, width, height, 4, rgba, width * 4) != 0;
}
