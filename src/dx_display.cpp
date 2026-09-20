#include "dx_display.h"
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr DXGI_FORMAT kFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT kUiFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

// Fullscreen composite of the UI texture: unpremultiply, sRGB -> linear,
// lift to SDR white, premultiplied blend over the back buffer.
const char* kComposeHlsl = R"HLSL(
Texture2D uiTex : register(t0);
SamplerState samp : register(s0);
cbuffer C : register(b0) { float whiteScale; float3 pad; };
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VSOut VSMain(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
float3 toLinear(float3 c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float4 PSMain(VSOut i) : SV_TARGET {
    float4 t = uiTex.Sample(samp, i.uv);
    float a = t.a;
    float3 c = t.rgb / max(a, 1e-4);
    c = toLinear(saturate(c)) * a * whiteScale;
    return float4(c, a);
}
)HLSL";

// Minimal descriptor allocator for Dear ImGui's DX12 backend.
struct SrvAllocator {
    ID3D12DescriptorHeap* heap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuStart{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpuStart{};
    unsigned stride = 0;
    std::vector<int> free;
    void init(ID3D12Device* dev, ID3D12DescriptorHeap* h, int count) {
        heap = h;
        cpuStart = h->GetCPUDescriptorHandleForHeapStart();
        gpuStart = h->GetGPUDescriptorHandleForHeapStart();
        stride = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        free.clear();
        for (int i = count - 1; i >= 0; --i) free.push_back(i);
    }
    void alloc(D3D12_CPU_DESCRIPTOR_HANDLE* cpu, D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
        const int i = free.back();
        free.pop_back();
        cpu->ptr = cpuStart.ptr + (SIZE_T)i * stride;
        gpu->ptr = gpuStart.ptr + (UINT64)i * stride;
    }
    void release(D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE) {
        free.push_back((int)((cpu.ptr - cpuStart.ptr) / stride));
    }
};
SrvAllocator g_srv;

bool fail(const char* what, HRESULT hr) {
    std::fprintf(stderr, "D3D12: %s failed (0x%08lx)\n", what, (unsigned long)hr);
    return false;
}

}  // namespace

DxDisplay::~DxDisplay() { shutdown(); }

bool DxDisplay::init(HWND hwnd, int width, int height) {
    hwnd_ = hwnd;
    width_ = width;
    height_ = height;
    HRESULT hr;
#ifndef NDEBUG
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
    }
#endif
    ComPtr<IDXGIFactory6> factory;
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return fail("CreateDXGIFactory2", hr);
    // The adapter with the most video memory = the RTX, which is where CUDA
    // lives; the shared buffer must be on the same GPU.
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;
         factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&device_))))
            break;
    }
    if (!device_) return fail("D3D12CreateDevice", E_FAIL);

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) return fail("CreateCommandQueue", hr);

    BOOL allowTearing = FALSE;
    if (SUCCEEDED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing,
                                               sizeof allowTearing)))
        tearing_ = allowTearing != 0;

    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = (UINT)width;
    sd.Height = (UINT)height;
    sd.Format = kFormat;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kFrames;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    sd.Flags = tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ComPtr<IDXGISwapChain1> sc1;
    hr = factory->CreateSwapChainForHwnd(queue_.Get(), hwnd, &sd, nullptr, nullptr, &sc1);
    if (FAILED(hr)) return fail("CreateSwapChainForHwnd", hr);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    hr = sc1.As(&swapchain_);
    if (FAILED(hr)) return fail("IDXGISwapChain3", hr);
    // FP16 + this colour space is scRGB: linear, 1.0 = SDR white reference
    // (80 nits), values above 1 are HDR when the display allows it.
    swapchain_->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);

    D3D12_DESCRIPTOR_HEAP_DESC rd{};
    rd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rd.NumDescriptors = kRtvCount;
    hr = device_->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&rtvHeap_));
    if (FAILED(hr)) return fail("CreateDescriptorHeap RTV", hr);
    rtvStride_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 64;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&srvHeap_));
    if (FAILED(hr)) return fail("CreateDescriptorHeap SRV", hr);
    g_srv.init(device_.Get(), srvHeap_.Get(), 64);

    for (int i = 0; i < kFrames; ++i) {
        hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocators_[i]));
        if (FAILED(hr)) return fail("CreateCommandAllocator", hr);
    }
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(),
                                    nullptr, IID_PPV_ARGS(&cmdList_));
    if (FAILED(hr)) return fail("CreateCommandList", hr);
    cmdList_->Close();

    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) return fail("CreateFence", hr);
    fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    if (!createSwapchainResources()) return false;
    if (!createSharedBuffer()) return false;
    if (!createUiTexture()) return false;
    if (!createComposePipeline()) return false;
    return true;
}

bool DxDisplay::createUiTexture() {
    uiTex_.Reset();
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = (UINT64)width_;
    rd.Height = (UINT)height_;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = kUiFormat;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{};
    cv.Format = kUiFormat;
    HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
                                                  IID_PPV_ARGS(&uiTex_));
    if (FAILED(hr)) return fail("CreateCommittedResource ui", hr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += (SIZE_T)kFrames * rtvStride_;
    device_->CreateRenderTargetView(uiTex_.Get(), nullptr, rtv);
    if (!uiSrvAllocated_) {
        g_srv.alloc(&uiSrvCpu_, &uiSrvGpu_);
        uiSrvAllocated_ = true;
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = kUiFormat;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(uiTex_.Get(), &sv, uiSrvCpu_);
    return true;
}

bool DxDisplay::createComposePipeline() {
    ComPtr<ID3DBlob> vs, ps, err;
    HRESULT hr = D3DCompile(kComposeHlsl, strlen(kComposeHlsl), "compose", nullptr, nullptr,
                            "VSMain", "vs_5_0", 0, 0, &vs, &err);
    if (FAILED(hr)) {
        if (err) std::fprintf(stderr, "%s\n", (const char*)err->GetBufferPointer());
        return fail("D3DCompile VS", hr);
    }
    hr = D3DCompile(kComposeHlsl, strlen(kComposeHlsl), "compose", nullptr, nullptr, "PSMain",
                    "ps_5_0", 0, 0, &ps, &err);
    if (FAILED(hr)) {
        if (err) std::fprintf(stderr, "%s\n", (const char*)err->GetBufferPointer());
        return fail("D3DCompile PS", hr);
    }

    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = 0;
    range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &range;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.ShaderRegister = 0;
    params[1].Constants.Num32BitValues = 4;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC samp{};
    samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samp.AddressU = samp.AddressV = samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.ShaderRegister = 0;
    samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = params;
    rsd.NumStaticSamplers = 1;
    rsd.pStaticSamplers = &samp;
    ComPtr<ID3DBlob> rsBlob;
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &err);
    if (FAILED(hr)) return fail("SerializeRootSignature", hr);
    hr = device_->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                      IID_PPV_ARGS(&composeRs_));
    if (FAILED(hr)) return fail("CreateRootSignature", hr);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = composeRs_.Get();
    pd.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pd.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pd.BlendState.RenderTarget[0].BlendEnable = TRUE;
    pd.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    pd.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    pd.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    pd.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    pd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    pd.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.DepthStencilState.DepthEnable = FALSE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = kFormat;
    pd.SampleDesc.Count = 1;
    hr = device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&composePso_));
    if (FAILED(hr)) return fail("CreateGraphicsPipelineState compose", hr);
    return true;
}

bool DxDisplay::createSwapchainResources() {
    for (int i = 0; i < kFrames; ++i) {
        HRESULT hr = swapchain_->GetBuffer((UINT)i, IID_PPV_ARGS(&backBuffers_[i]));
        if (FAILED(hr)) return fail("GetBuffer", hr);
        D3D12_CPU_DESCRIPTOR_HANDLE h = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        h.ptr += (SIZE_T)i * rtvStride_;
        device_->CreateRenderTargetView(backBuffers_[i].Get(), nullptr, h);
    }
    return true;
}

void DxDisplay::releaseSwapchainResources() {
    for (int i = 0; i < kFrames; ++i) backBuffers_[i].Reset();
}

bool DxDisplay::createSharedBuffer() {
    releaseSharedBuffer();
    // Row pitch must satisfy D3D12's 256-byte placed-footprint alignment.
    rowPitch_ = ((width_ * 8) + 255) & ~255;
    sharedSize_ = (size_t)rowPitch_ * (size_t)height_;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sharedSize_;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(&shared_));
    if (FAILED(hr)) return fail("CreateCommittedResource shared", hr);
    hr = device_->CreateSharedHandle(shared_.Get(), nullptr, GENERIC_ALL, nullptr,
                                     &sharedHandle_);
    if (FAILED(hr)) return fail("CreateSharedHandle", hr);
    return true;
}

void DxDisplay::releaseSharedBuffer() {
    if (sharedHandle_) {
        CloseHandle(sharedHandle_);
        sharedHandle_ = nullptr;
    }
    shared_.Reset();
}

void DxDisplay::waitForGpu() {
    if (!fence_ || fenceValue_ == 0) return;
    if (fence_->GetCompletedValue() < fenceValue_) {
        fence_->SetEventOnCompletion(fenceValue_, fenceEvent_);
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

bool DxDisplay::resize(int width, int height) {
    if (width == width_ && height == height_) return false;
    if (width <= 0 || height <= 0) return false;
    waitForGpu();
    releaseSwapchainResources();
    width_ = width;
    height_ = height;
    HRESULT hr = swapchain_->ResizeBuffers(kFrames, (UINT)width, (UINT)height, kFormat,
                                           tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    if (FAILED(hr)) {
        fail("ResizeBuffers", hr);
        return true;
    }
    createSwapchainResources();
    createSharedBuffer();
    createUiTexture();
    return true;
}

bool DxDisplay::present(ImDrawData* drawData, bool vsync, float whiteScale) {
    const UINT bi = swapchain_->GetCurrentBackBufferIndex();
    // Make sure this frame's allocator is free.
    if (frameFence_[bi] != 0 && fence_->GetCompletedValue() < frameFence_[bi]) {
        fence_->SetEventOnCompletion(frameFence_[bi], fenceEvent_);
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
    allocators_[bi]->Reset();
    cmdList_->Reset(allocators_[bi].Get(), nullptr);

    ID3D12Resource* bb = backBuffers_[bi].Get();
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = bb;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    cmdList_->ResourceBarrier(1, &b);

    // Shared buffer -> back buffer.
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = shared_.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = kFormat;
    src.PlacedFootprint.Footprint.Width = (UINT)width_;
    src.PlacedFootprint.Footprint.Height = (UINT)height_;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch_;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = bb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    cmdList_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    cmdList_->ResourceBarrier(1, &b);

    ID3D12DescriptorHeap* heaps[] = {srvHeap_.Get()};
    cmdList_->SetDescriptorHeaps(1, heaps);

    // UI into its own 8-bit texture (ImGui outputs sRGB-encoded colours).
    D3D12_RESOURCE_BARRIER ub{};
    ub.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    ub.Transition.pResource = uiTex_.Get();
    ub.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ub.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    ub.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    cmdList_->ResourceBarrier(1, &ub);
    D3D12_CPU_DESCRIPTOR_HANDLE uiRtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    uiRtv.ptr += (SIZE_T)kFrames * rtvStride_;
    const float clear[4] = {0.f, 0.f, 0.f, 0.f};
    cmdList_->ClearRenderTargetView(uiRtv, clear, 0, nullptr);
    cmdList_->OMSetRenderTargets(1, &uiRtv, FALSE, nullptr);
    if (drawData && imguiUp_) ImGui_ImplDX12_RenderDrawData(drawData, cmdList_.Get());
    ub.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    ub.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cmdList_->ResourceBarrier(1, &ub);

    // Composite the UI over the image in linear light.
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += (SIZE_T)bi * rtvStride_;
    cmdList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT vp{0.f, 0.f, (float)width_, (float)height_, 0.f, 1.f};
    D3D12_RECT sc{0, 0, width_, height_};
    cmdList_->RSSetViewports(1, &vp);
    cmdList_->RSSetScissorRects(1, &sc);
    cmdList_->SetPipelineState(composePso_.Get());
    cmdList_->SetGraphicsRootSignature(composeRs_.Get());
    cmdList_->SetGraphicsRootDescriptorTable(0, uiSrvGpu_);
    const float consts[4] = {whiteScale, 0.f, 0.f, 0.f};
    cmdList_->SetGraphicsRoot32BitConstants(1, 4, consts, 0);
    cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList_->DrawInstanced(3, 1, 0, 0);

    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList_->ResourceBarrier(1, &b);
    cmdList_->Close();
    ID3D12CommandList* lists[] = {cmdList_.Get()};
    queue_->ExecuteCommandLists(1, lists);

    HRESULT hr;
    if (vsync) {
        hr = swapchain_->Present(1, 0);
    } else {
        hr = swapchain_->Present(0, tearing_ ? DXGI_PRESENT_ALLOW_TEARING : 0);
    }
    ++fenceValue_;
    queue_->Signal(fence_.Get(), fenceValue_);
    frameFence_[bi] = fenceValue_;
    if (FAILED(hr)) return fail("Present", hr);
    return true;
}

bool DxDisplay::imguiInit() {
    ImGui_ImplDX12_InitInfo info{};
    info.Device = device_.Get();
    info.CommandQueue = queue_.Get();
    info.NumFramesInFlight = kFrames;
    info.RTVFormat = kUiFormat;
    info.DSVFormat = DXGI_FORMAT_UNKNOWN;
    info.SrvDescriptorHeap = srvHeap_.Get();
    info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                                   D3D12_GPU_DESCRIPTOR_HANDLE* gpu) { g_srv.alloc(cpu, gpu); };
    info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                  D3D12_GPU_DESCRIPTOR_HANDLE gpu) { g_srv.release(cpu, gpu); };
    imguiUp_ = ImGui_ImplDX12_Init(&info);
    return imguiUp_;
}

void DxDisplay::imguiNewFrame() {
    if (imguiUp_) ImGui_ImplDX12_NewFrame();
}

void DxDisplay::imguiShutdown() {
    if (imguiUp_) {
        waitForGpu();
        ImGui_ImplDX12_Shutdown();
        imguiUp_ = false;
    }
}

void DxDisplay::shutdown() {
    if (!device_) return;
    waitForGpu();
    imguiShutdown();
    releaseSharedBuffer();
    releaseSwapchainResources();
    uiTex_.Reset();
    composePso_.Reset();
    composeRs_.Reset();
    if (fenceEvent_) {
        CloseHandle(fenceEvent_);
        fenceEvent_ = nullptr;
    }
    fence_.Reset();
    cmdList_.Reset();
    for (int i = 0; i < kFrames; ++i) allocators_[i].Reset();
    srvHeap_.Reset();
    rtvHeap_.Reset();
    swapchain_.Reset();
    queue_.Reset();
    device_.Reset();
}
