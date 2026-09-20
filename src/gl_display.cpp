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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[4] = {0.f, 0.f, 0.f, 1.f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
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

bool GlDisplay::readPixels(unsigned* rgba, int count) const {
    if (!pbo_ || count < w_ * h_) return false;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)w_ * h_ * 4, rgba);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return true;
}

void GlDisplay::draw(float u0, float v0, float u1, float v1) const {
    if (!tex_) return;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    dl->AddImage((ImTextureID)(uintptr_t)tex_, ImVec2(0, 0), ImVec2((float)w_, (float)h_),
                 ImVec2(u0, v0), ImVec2(u1, v1));
}
