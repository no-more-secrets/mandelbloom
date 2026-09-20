#pragma once

// Owns the GL texture and pixel unpack buffer that show the rendered image.
class GlDisplay {
public:
    ~GlDisplay();
    // Allocate or reallocate for a new size. Returns true if size changed.
    bool resize(int width, int height);
    // Copy PBO contents into the texture (GPU-side copy).
    void upload();
    // Queue the texture as the ImGui background covering the window.
    void draw() const;

    unsigned pbo() const { return pbo_; }
    unsigned texture() const { return tex_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    void destroy();
    unsigned tex_ = 0;
    unsigned pbo_ = 0;
    int w_ = 0, h_ = 0;
};
