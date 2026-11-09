// SPDX-License-Identifier: MIT
#include "TimelineProcessor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "FrameDigest.h"

namespace resolve_dlss5 {
TimelineProcessor::TimelineProcessor(std::size_t budget) : budget_(budget) {}
TimelineProcessor::Signature TimelineProcessor::signature(double time, const CpuFrame& frame) {
    if (!std::isfinite(time) || frame.width <= 0 || frame.height <= 0 || frame.width > 16384 ||
        frame.height > 16384 ||
        frame.pixels.size() != static_cast<std::size_t>(frame.width) * frame.height * 4)
        throw std::invalid_argument("Current frame has an invalid time or extent");
    Signature value;
    value.time = time;
    value.width = frame.width;
    value.height = frame.height;
    value.settings = frame.settings;
    value.flow = frame.flow;
    if (value.flow.external) {
        value.flow.method = FlowMethod::None;
        value.flow.amdQuality = 1;
        value.flow.nvidiaQuality = 3;
    } else {
        value.flow.xChannel = 0;
        value.flow.yChannel = 1;
        value.flow.units = FlowUnits::Pixels;
        value.flow.yUp = false;
        value.flow.scaleX = value.flow.scaleY = 1;
        if (value.flow.method != FlowMethod::Amd) value.flow.amdQuality = 1;
        if (value.flow.method != FlowMethod::Nvidia) value.flow.nvidiaQuality = 3;
    }
    // Final correction controls do not alter the neural proxy or its history.
    value.settings.detail = {};
    value.settings.hdrTransferStrength = 1;
    for (int i = std::max(0, value.settings.passCount - 1); i < 2; ++i)
        value.settings.additionalPasses[i] = {};
    fingerprintFloats(frame.pixels, value.digest.data());
    if (!frame.referencePixels.empty() && frame.referencePixels.size() != frame.pixels.size())
        throw std::invalid_argument("Invalid optical flow reference image");
    if (!frame.motion.empty() &&
        frame.motion.size() != static_cast<std::size_t>(frame.width) * frame.height * 2)
        throw std::invalid_argument("Invalid dense external motion extent");
    auto fingerprint = [&](const std::vector<float>& pixels, std::size_t offset) {
        if (!pixels.empty()) fingerprintFloats(pixels, value.digest.data() + offset);
    };
    fingerprint(frame.referencePixels, 32);
    fingerprint(frame.motion, 64);
    return value;
}
void TimelineProcessor::invalidate() {
    committed_.reset();
    cached_.reset();
    stats_.cachedBytes = 0;
}
TimelineProcessor::Output TimelineProcessor::render(double time, const CpuFrame& frame,
                                                    const Evaluate& evaluate, const Reset& reset,
                                                    const std::function<bool()>& abort,
                                                    TimelineDomain domain, double frameStep) {
    try {
        if (abort && abort()) throw std::runtime_error("Streaming render cancelled");
        if (!std::isfinite(frameStep) || frameStep <= 0)
            throw std::invalid_argument("Invalid streaming frame step");
        using Clock = std::chrono::steady_clock;
        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        stats_.signatureMs = stats_.evaluateMs = stats_.validateMs = 0;
        const auto signatureStart = Clock::now();
        const Signature current = signature(time, frame);
        stats_.signatureMs = ms(signatureStart, Clock::now());
        ++stats_.inputChecks;
        const bool sameDomain = committed_ && domain == domain_ && frameStep == frameStep_;
        if (sameDomain && cached_ && current == *committed_) {
            ++stats_.cacheHits;
            return cached_;
        }
        const bool continuous =
            sameDomain && std::abs(time - committed_->time - frameStep) <= 1e-6 &&
            current.width == committed_->width && current.height == committed_->height &&
            current.settings == committed_->settings && current.flow == committed_->flow &&
            (frame.referencePixels.empty() ||
             std::equal(current.digest.begin() + 32, current.digest.begin() + 64,
                        committed_->digest.begin()));
        const bool resetHistory = !continuous;
        if (resetHistory) ++stats_.historyResets;
        // Evaluate owns the actual Reset flag. Do not destroy/recreate Feature
        // resources for every seek when an ordinary model-history reset suffices.
        const auto evaluateStart = Clock::now();
        auto result = std::make_shared<std::vector<float>>(evaluate(frame, resetHistory));
        const auto validateStart = Clock::now();
        stats_.evaluateMs = ms(evaluateStart, validateStart);
        ++stats_.evaluatedFrames;
        if (result->size() != frame.pixels.size())
            throw std::runtime_error("Incomplete neural frame output");
        std::atomic<bool> finite{true};
        const std::size_t rowFloats = static_cast<std::size_t>(frame.width) * 4;
        parallelRows(frame.height, static_cast<std::size_t>(frame.width) * frame.height,
                     [&](int y) noexcept {
                         const float* row = result->data() + static_cast<std::size_t>(y) * rowFloats;
                         for (std::size_t i = 0; i < rowFloats; ++i)
                             if (!std::isfinite(row[i])) {
                                 finite.store(false, std::memory_order_relaxed);
                                 return;
                             }
                     });
        if (!finite.load()) throw std::runtime_error("Non-finite neural frame output");
        stats_.validateMs = ms(validateStart, Clock::now());
        if (abort && abort()) throw std::runtime_error("Streaming render cancelled");
        committed_ = current;
        domain_ = domain;
        frameStep_ = frameStep;
        const std::size_t bytes = result->size() * sizeof(float);
        cached_ = bytes <= budget_ ? result : Output{};
        stats_.cachedBytes = cached_ ? bytes : 0;
        return result;
    } catch (...) {
        invalidate();
        reset();
        throw;
    }
}
}  // namespace resolve_dlss5
