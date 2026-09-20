#include "video_encoder.h"
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

std::string quote(const std::string& s) {
    if (!s.empty() && s.find_first_of(" \t\"") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) {
        if (c == '"') q += "\\\"";
        else q += c;
    }
    return q + "\"";
}

}  // namespace

VideoEncoder::~VideoEncoder() { cancel(); }

bool VideoEncoder::start(const VideoEncodeSettings& s) {
    if (proc_) return false;
    err_.clear();
    const std::string codec = s.codec.empty() ? "hevc_nvenc" : s.codec;
    const char* colour = s.hdr ? " -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc"
                               : " -color_primaries bt709 -color_trc bt709 -colorspace bt709";
    char buf[512];
    std::string cmd = quote(s.ffmpeg) + " -hide_banner -loglevel error -y -f rawvideo";
    std::snprintf(buf, sizeof buf, " -pix_fmt %s -s %dx%d -r %.6g", s.hdr ? "p010le" : "yuv420p",
                  s.width, s.height, s.fps);
    cmd += buf;
    cmd += colour;
    cmd += " -color_range tv -i - -an -c:v " + codec;
    if (codec.find("nvenc") != std::string::npos) {
        // Without a maxrate, hevc_nvenc caps the stream at a small default
        // bitrate and cq has no effect. 0.25 bit/pixel/frame, 30..200 Mbps.
        const double mbps = std::min(200.0, std::max(30.0, (double)s.width * s.height * s.fps * 0.25 / 1e6));
        std::snprintf(buf, sizeof buf, " -preset p5 -tune hq -rc vbr -cq %d -b:v 0 -maxrate %.0fM -bufsize %.0fM",
                      s.cq, mbps, 2.0 * mbps);
        cmd += buf;
        if (codec == "hevc_nvenc") cmd += s.hdr ? " -profile:v main10" : " -profile:v main";
    } else {
        std::snprintf(buf, sizeof buf, " -crf %d", s.cq);
        cmd += buf;
    }
    cmd += s.hdr ? " -pix_fmt p010le" : " -pix_fmt yuv420p";
    cmd += colour;
    cmd += " -color_range tv";
    if (codec.find("hevc") != std::string::npos || codec.find("265") != std::string::npos)
        cmd += " -tag:v hvc1";
    cmd += " -movflags +faststart " + quote(s.outPath);

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 22)) {
        err_ = "CreatePipe failed";
        return false;
    }
    SetHandleInformation(wr, HANDLE_FLAG_INHERIT, 0);
    HANDLE log = CreateFileA(s.logPath.empty() ? "NUL" : s.logPath.c_str(), GENERIC_WRITE,
                             FILE_SHARE_READ, &sa, s.logPath.empty() ? OPEN_EXISTING : CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOA si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = rd;
    si.hStdOutput = log;
    si.hStdError = log;
    PROCESS_INFORMATION pi{};
    std::vector<char> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back('\0');
    const BOOL ok = CreateProcessA(nullptr, cmdBuf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                   nullptr, nullptr, &si, &pi);
    CloseHandle(rd);
    if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
    if (!ok) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "CreateProcess failed (%lu): is ffmpeg on PATH?",
                      (unsigned long)GetLastError());
        err_ = msg;
        CloseHandle(wr);
        return false;
    }
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    pipeW_ = wr;
    cmd_ = cmd;
    stop_ = false;
    failed_ = false;
    pending_ = nullptr;
    writer_ = std::thread([this] { writerLoop(); });
    return true;
}

void VideoEncoder::writerLoop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [this] { return stop_ || pending_ != nullptr; });
        if (pending_ == nullptr) break;  // stop with nothing left to write
        const char* p = (const char*)pending_;
        size_t n = pendingBytes_;
        lk.unlock();
        bool ok = true;
        while (n > 0) {
            DWORD w = 0;
            const DWORD chunk = (DWORD)std::min<size_t>(n, (size_t)1 << 24);
            if (!WriteFile((HANDLE)pipeW_, p, chunk, &w, nullptr) || w == 0) {
                ok = false;
                break;
            }
            p += w;
            n -= w;
        }
        lk.lock();
        pending_ = nullptr;
        if (!ok) failed_ = true;
        cv_.notify_all();
    }
}

bool VideoEncoder::submit(const void* data, size_t bytes) {
    if (!proc_) return false;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return pending_ == nullptr; });
    if (failed_) {
        err_ = "pipe write failed (ffmpeg exited?)";
        return false;
    }
    pending_ = data;
    pendingBytes_ = bytes;
    cv_.notify_all();
    return true;
}

int VideoEncoder::finish() {
    if (!proc_) return -1;
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return pending_ == nullptr; });
        stop_ = true;
        cv_.notify_all();
    }
    if (writer_.joinable()) writer_.join();
    CloseHandle((HANDLE)pipeW_);
    pipeW_ = nullptr;
    WaitForSingleObject((HANDLE)proc_, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess((HANDLE)proc_, &code);
    CloseHandle((HANDLE)proc_);
    proc_ = nullptr;
    if (failed_ && code == 0) code = 1;
    return (int)code;
}

void VideoEncoder::cancel() {
    if (!proc_) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        cv_.notify_all();
    }
    // A blocked WriteFile returns once the read end goes away with ffmpeg.
    TerminateProcess((HANDLE)proc_, 1);
    WaitForSingleObject((HANDLE)proc_, 5000);
    if (writer_.joinable()) writer_.join();
    CloseHandle((HANDLE)pipeW_);
    pipeW_ = nullptr;
    CloseHandle((HANDLE)proc_);
    proc_ = nullptr;
}
