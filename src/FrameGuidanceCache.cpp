// SPDX-License-Identifier: MIT
#include "FrameGuidanceCache.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "FrameDigest.h"
namespace resolve_dlss5 {
FlowSettings activeFlowSettings(FlowSettings s) {
    if (s.external) {
        s.method = FlowMethod::None;
        s.amdQuality = 1;
        s.nvidiaQuality = 3;
    } else {
        s.xChannel = 0;
        s.yChannel = 1;
        s.units = FlowUnits::Pixels;
        s.yUp = false;
        s.scaleX = s.scaleY = 1;
        if (s.method != FlowMethod::Amd) s.amdQuality = 1;
        if (s.method != FlowMethod::Nvidia) s.nvidiaQuality = 3;
    }
    return s;
}
FlowSettings guidanceDemand(FlowSettings s, bool temporalMotion, FlowMethod fallback) {
    if (!s.external && s.method == FlowMethod::None && temporalMotion) {
        if (fallback != FlowMethod::Amd && fallback != FlowMethod::Nvidia)
            throw std::invalid_argument("Temporal motion requires AMDOF or NVOF");
        s.method = fallback;
    }
    return activeFlowSettings(s);
}
FrameGuidanceCache::Output FrameGuidanceCache::get(
    double time, int width, int height, const Feature18Settings& color, FlowSettings flow,
    std::span<const float> current, std::span<const float> reference,
    std::span<const float> external, const std::function<MotionField()>& produce) {
    const auto count = static_cast<std::size_t>(std::max(width, 0)) * std::max(height, 0);
    if (!std::isfinite(time) || width <= 0 || height <= 0 || count > 16384U * 16384U ||
        current.size() != count * 4 || (!reference.empty() && reference.size() != current.size()) ||
        (!external.empty() && external.size() != count * 2))
        throw std::invalid_argument("Invalid guidance cache input");
    Key key;
    key.time = time;
    key.width = width;
    key.height = height;
    key.flow = activeFlowSettings(flow);
    key.codec.inputEncoding = color.inputEncoding;
    key.codec.premultiplied = color.premultiplied;
    key.codec.paperWhiteScale = color.paperWhiteScale;
    key.codec.referenceWhiteNits = color.referenceWhiteNits;
    key.codec.peakNits = color.peakNits;
    auto hash = [&](std::span<const float> values, int offset) {
        if (!values.empty()) fingerprintFloats(values, key.digest.data() + offset);
    };
    hash(current, 0);
    hash(reference, 32);
    hash(external, 64);
    if (key_ && *key_ == key && output_) return output_;
    // One dense field; the pointer stays immutable through every consumer.
    if (count * sizeof(float) * 2 > 256U * 1024U * 1024U)
        throw std::runtime_error("Guidance cache exceeds its bounded memory budget");
    try {
        auto output = std::make_shared<const MotionField>(produce());
        if (output->size() != count * 2) throw std::runtime_error("Incomplete shared motion field");
        std::atomic<bool> finite{true};
        const std::size_t rowFloats = static_cast<std::size_t>(width) * 2;
        parallelRows(height, count, [&](int y) noexcept {
            const float* row = output->data() + static_cast<std::size_t>(y) * rowFloats;
            for (std::size_t i = 0; i < rowFloats; ++i)
                if (!std::isfinite(row[i])) {
                    finite.store(false, std::memory_order_relaxed);
                    return;
                }
        });
        if (!finite.load()) throw std::runtime_error("Non-finite shared motion field");
        output_ = std::move(output);
        key_ = key;
        ++revision_;
        ++productions_;
        return output_;
    } catch (...) {
        invalidate();
        throw;
    }
}
}  // namespace resolve_dlss5
