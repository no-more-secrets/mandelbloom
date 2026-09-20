#pragma once
// Direct3D 12 presentation: an FP16 scRGB swapchain (HDR on HDR displays,
// plain linear on SDR ones), a shared buffer that CUDA writes RGBA16F pixels
// into, and Dear ImGui on top. Windows only.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>

struct ImDrawData;

class DxDisplay {
public:
    ~DxDisplay();
    bool init(HWND hwnd, int width, int height);
    void shutdown();
    // Recreate the swapchain buffers and the shared buffer. Returns true if
    // the size changed (the caller must re-import the shared buffer).
    bool resize(int width, int height);

    // Shared buffer CUDA writes into: RGBA16F rows of rowPitch() bytes.
    HANDLE sharedHandle() const { return sharedHandle_; }
    size_t sharedSize() const { return sharedSize_; }
    int rowPitch() const { return rowPitch_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Block until the GPU finished the previous frame's copy out of the
    // shared buffer, so CUDA may overwrite it.
    void waitForGpu();
    // Copy the shared buffer to the back buffer, draw the UI, present.
    bool present(ImDrawData* drawData, bool vsync);

    bool imguiInit();
    void imguiNewFrame();
    void imguiShutdown();

private:
    bool createSwapchainResources();
    void releaseSwapchainResources();
    bool createSharedBuffer();
    void releaseSharedBuffer();

    static constexpr int kFrames = 3;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapchain_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> backBuffers_[kFrames];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocators_[kFrames];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    uint64_t fenceValue_ = 0;
    uint64_t frameFence_[kFrames] = {};
    Microsoft::WRL::ComPtr<ID3D12Resource> shared_;
    HANDLE sharedHandle_ = nullptr;
    size_t sharedSize_ = 0;
    int rowPitch_ = 0;
    int width_ = 0, height_ = 0;
    HWND hwnd_ = nullptr;
    bool tearing_ = false;
    unsigned rtvStride_ = 0;
    bool imguiUp_ = false;
};
