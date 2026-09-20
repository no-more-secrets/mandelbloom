#include <glad/glad.h>
#include <imgui.h>
#include <cstdint>
#include "gl_display.h"

GlDisplay::~GlDisplay() { destroy(); }

void GlDisplay::destroy() {
    if (tex_) glDeleteTextures(1, &tex_);
    if (pbo_) glDeleteBuffers(1, &pbo_);
    tex_ = pbo_ = 0;
    w_ = h_ = 0;
}

bool GlDisplay::resize(int width, int height) {
    if (width == w_ && height == h_) return false;
    destroy();
    if (width <= 0 || height <= 0) return true;
    w_ = width;
    h_ = height;

    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)w_ * h_ * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w_, h_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void GlDisplay::upload() {
    if (!tex_ || !pbo_) return;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GlDisplay::draw() const {
    if (!tex_) return;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    dl->AddImage((ImTextureID)(uintptr_t)tex_, ImVec2(0, 0), ImVec2((float)w_, (float)h_));
}
