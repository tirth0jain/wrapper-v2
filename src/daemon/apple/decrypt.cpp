#include "apple/decrypt.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "apple/fps_cert.inc"
#include "apple/aarch64_sret_thunks.hpp"
#include "apple/loader.hpp"
#include "apple/runtime.hpp"

namespace wrapper::apple {

namespace {

constexpr char kPrefetchAdam[] = "0";

// zhaarey/wrapper main.c getKdContext:
// - Never destroys SVFootHillPContext (stack POD shared_ptr, no refcount decrement).
// - preshareCtx: first successful adam=="0" path caches kd; later adam=="0" skips
//   getPersistentKey entirely (no URI check in upstream).
// - This wrapper also caches regular track kd contexts by adam_id + key URI so
//   batch decrypt requests do not repeatedly process the same CKC.
//
// Calling shared_ptr_SVFootHillPContext_dtor breaks kd / Apple's FootHill state.

std::mutex g_kd_cache_mu;
std::unordered_map<std::string, void*> g_kd_cache;

std::string kd_cache_key(const std::string& adam_id, const std::string& key_uri) {
    std::string key;
    key.reserve(adam_id.size() + 1 + key_uri.size());
    key.append(adam_id);
    key.push_back('\n');
    key.append(key_uri);
    return key;
}

void* find_cached_kd(const std::string& adam_id, const std::string& key_uri) {
    std::lock_guard<std::mutex> lock(g_kd_cache_mu);
    auto it = g_kd_cache.find(kd_cache_key(adam_id, key_uri));
    if (it == g_kd_cache.end()) return nullptr;
    return it->second;
}

void store_cached_kd(const std::string& adam_id, const std::string& key_uri, void* kd) {
    std::lock_guard<std::mutex> lock(g_kd_cache_mu);
    g_kd_cache[kd_cache_key(adam_id, key_uri)] = kd;
}

void erase_cached_kd(const std::string& adam_id, const std::string& key_uri) {
    std::lock_guard<std::mutex> lock(g_kd_cache_mu);
    g_kd_cache.erase(kd_cache_key(adam_id, key_uri));
}

}  // namespace

DecryptResult decrypt_samples(const Loader& loader,
                              Runtime&      runtime,
                              std::string   adam_id,
                              std::string   key_uri,
                              std::vector<std::vector<std::uint8_t>> ciphertexts) {
    DecryptResult out;
    if (!loader.ok() || !loader.fps_decrypt_available()) {
        out.error = "FPS decrypt chain not loaded";
        return out;
    }
    if (!runtime.playback_ready()) {
        out.error = "playback stack not ready";
        return out;
    }
    if (adam_id.empty() || key_uri.empty()) {
        out.error = "adam_id and uri are required";
        return out;
    }
    if (ciphertexts.empty()) {
        out.error = "at least one sample required";
        return out;
    }

    const Symbols& s  = loader.sym();
    void*          fh = runtime.foothill_session();

    auto decrypt_once = [&](bool allow_cache,
                            std::vector<std::vector<std::uint8_t>> chunks,
                            std::string* error) -> DecryptResult {
        DecryptResult attempt;
        void* kd = nullptr;

        if (allow_cache) {
            kd = find_cached_kd(adam_id, key_uri);
        }

        if (kd == nullptr && adam_id == kPrefetchAdam) {
            *error = "prefetch decrypt context is not cached; decrypt prefetch samples locally";
            return attempt;
        }

        if (kd == nullptr) {
            auto        default_id = abi::make_string_view(adam_id.c_str());
            auto        uri        = abi::make_string_view(key_uri.c_str());
            auto        key_format = abi::make_string_view("com.apple.streamingkeydelivery");
            auto        key_ver    = abi::make_string_view("1");
            auto        server_uri =
                abi::make_string_view("https://play.itunes.apple.com/WebObjects/MZPlay.woa/music/fps");
            auto        protocol   = abi::make_string_view("simplified");
            auto        fps_cert   = abi::make_string_view(kFpsCert);

            abi::shared_ptr persist{};
            loader.foot_hill_get_persistent_key(
                &persist, fh,
                &default_id, &uri, &key_format, &key_ver,
                &server_uri, &protocol, &fps_cert);

            if (persist.obj == nullptr) {
                *error = "getPersistentKey failed (key or lease?)";
                return attempt;
            }

            abi::shared_ptr sv_ctx{};
            aarch64_sret::svfoot_decrypt_context(&sv_ctx, fh, persist.obj,
                                                s.SVFootHillSessionCtrl_decryptContext);

            if (sv_ctx.obj == nullptr) {
                *error = "decryptContext failed";
                return attempt;
            }

            // Upstream main.c does TWO dereferences:
            //   void* p = *kdContext_method(ctx);   // *(void**) -> void*
            //   ... NfcRKVn(*(void**)p, ...)         // re-cast and deref again
            // i.e. fp_sample_decrypt receives **kdContext_method(ctx). Doing only
            // one deref passes the kd-handle struct pointer instead of the actual
            // engine state pointer; fp_sample_decrypt doesn't error but the
            // produced plaintext is garbage (audio plays back unplayable).
            void** kd_pp = s.SVFootHillPContext_kdContext(sv_ctx.obj);
            if (kd_pp == nullptr || *kd_pp == nullptr) {
                *error = "kdContext is null";
                return attempt;
            }
            kd = *reinterpret_cast<void**>(*kd_pp);
            if (kd == nullptr) {
                *error = "kdContext inner pointer is null";
                return attempt;
            }

            store_cached_kd(adam_id, key_uri, kd);

            // Intentionally no shared_ptr dtors — see block comment above.
            (void)persist;
            (void)sv_ctx;
        }

        // Parallel sample decryption (adapted from 5hojib/wrapper-v2 #5).
        // fp_sample_decrypt is called with the SAME cached kd context from
        // multiple threads; the context holds the FairPlay key schedule and
        // the call is per-sample, so this is safe in practice. Thread count is
        // capped at 4 so the shared kd context is never hammered harder than
        // needed, and each index is handled by exactly one thread (stride t).
        attempt.plaintexts.resize(chunks.size());

        unsigned int hw = std::thread::hardware_concurrency();
        unsigned int num_threads = (hw == 0) ? 2 : std::min(hw, 4u);
        if (num_threads > chunks.size()) {
            num_threads = static_cast<unsigned int>(chunks.size());
        }
        if (num_threads < 1) {
            num_threads = 1;
        }

        std::vector<std::thread> threads;
        std::atomic<bool> decrypt_error{false};
        std::mutex error_mu;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                for (size_t i = t; i < chunks.size(); i += num_threads) {
                    if (decrypt_error.load(std::memory_order_relaxed)) {
                        break;
                    }
                    auto& chunk = chunks[i];
                    if (chunk.empty()) {
                        std::lock_guard<std::mutex> lock(error_mu);
                        if (!decrypt_error.exchange(true)) {
                            *error = "empty sample";
                        }
                        break;
                    }
                    const long status = s.fp_sample_decrypt(kd, 5u, chunk.data(), chunk.data(), chunk.size());
                    if (status < 0) {
                        std::lock_guard<std::mutex> lock(error_mu);
                        if (!decrypt_error.exchange(true)) {
                            *error = "FPS sample decrypt failed status=" + std::to_string(status);
                            std::fprintf(stderr, "decrypt: fp_sample_decrypt failed status=%ld\n", status);
                        }
                        break;
                    }
                    attempt.plaintexts[i] = std::move(chunk);
                }
            });
        }
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        if (decrypt_error.load()) {
            attempt.plaintexts.clear();
            return attempt;
        }

        attempt.ok = true;
        return attempt;
    };

    std::string first_error;
    try {
        out = decrypt_once(true, ciphertexts, &first_error);
    } catch (const std::exception& e) {
        first_error = e.what();
    } catch (...) {
        first_error = "native FPS decrypt threw an unknown exception";
    }
    if (out.ok) return out;

    erase_cached_kd(adam_id, key_uri);
    out.error = "FPS decrypt failed";
    if (!first_error.empty()) out.error += " (first: " + first_error + ")";
    return out;
}

}  // namespace wrapper::apple
