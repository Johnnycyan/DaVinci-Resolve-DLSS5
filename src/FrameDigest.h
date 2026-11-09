// SPDX-License-Identifier: MIT
#pragma once
#include <Windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "ParallelRows.h"

namespace resolve_dlss5 {
// Frame fingerprint used for cache and history identity: SHA-256 over the
// per-chunk SHA-256 digests plus the total length. Chunks hash in parallel on
// the shared row pool. Digests are only ever compared with other digests from
// this function, so every fingerprint site must use it.
inline void fingerprintFloats(std::span<const float> values, unsigned char* digest32) {
    constexpr std::size_t kChunkBytes = 128U * 1024U;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    const std::size_t total = values.size_bytes();
    const std::size_t chunks = std::max<std::size_t>(1, (total + kChunkBytes - 1) / kChunkBytes);
    if (chunks > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("Image is too large to fingerprint");
    std::vector<unsigned char> digests(chunks * 32 + sizeof(std::uint64_t));
    std::atomic<bool> ok{true};
    parallelRows(static_cast<int>(chunks), values.size() / 4, [&](int index) noexcept {
        const std::size_t begin = static_cast<std::size_t>(index) * kChunkBytes;
        const std::size_t length = std::min(kChunkBytes, total - begin);
        if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                       const_cast<PUCHAR>(bytes ? bytes + begin : nullptr),
                       static_cast<ULONG>(length), digests.data() + index * 32, 32) < 0)
            ok.store(false, std::memory_order_relaxed);
    });
    const std::uint64_t length = total;
    std::memcpy(digests.data() + chunks * 32, &length, sizeof(length));
    if (!ok.load() || BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, digests.data(),
                                 static_cast<ULONG>(digests.size()), digest32, 32) < 0)
        throw std::runtime_error("Cannot fingerprint image");
}
}  // namespace resolve_dlss5
