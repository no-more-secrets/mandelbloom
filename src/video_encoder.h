#pragma once
// Pipes raw video frames into an external ffmpeg process (no console window)
// which encodes them with NVENC. One frame may be in flight on a writer
// thread while the next one is being produced.
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>

struct VideoEncodeSettings {
    std::string ffmpeg = "ffmpeg";  // executable, on PATH or a full path
    std::string outPath;
    int width = 0, height = 0;
    double fps = 60.0;
    bool hdr = false;   // p010le input, HEVC main10, PQ / BT.2020; else yuv420p, HEVC, BT.709
    int cq = 20;        // NVENC constant quality (lower = better, larger)
    std::string codec;  // empty: hevc_nvenc
    std::string logPath;  // ffmpeg's stderr
};

class VideoEncoder {
public:
    ~VideoEncoder();
    bool start(const VideoEncodeSettings& s);
    // Queue one frame. Blocks until the previous frame has been written, so
    // the previous buffer may be reused once this returns. `data` must stay
    // valid until the next submit() or finish().
    bool submit(const void* data, size_t bytes);
    // Close the pipe and wait for ffmpeg. Returns its exit code (0 = ok).
    int finish();
    // Stop writing and kill ffmpeg.
    void cancel();
    bool running() const { return proc_ != nullptr; }
    const std::string& commandLine() const { return cmd_; }
    const std::string& error() const { return err_; }

private:
    void writerLoop();
    void* proc_ = nullptr;   // HANDLE
    void* pipeW_ = nullptr;  // HANDLE: our end of ffmpeg's stdin
    std::string cmd_, err_;
    std::thread writer_;
    std::mutex mu_;
    std::condition_variable cv_;
    const void* pending_ = nullptr;
    size_t pendingBytes_ = 0;
    bool stop_ = false;
    bool failed_ = false;
};
