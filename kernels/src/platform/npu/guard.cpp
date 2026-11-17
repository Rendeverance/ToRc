// guard.cpp - runs a call into ggml inside a C++ try, so an exception thrown in ggml (a weight buffer the NPU driver
// will not map, say) comes back to the engine as an error and the CPU carries on: Rust cannot catch a C++ exception,
// and one reaching it ends the process. Built on Linux only (kernels/build.rs), where ggml and this file share the
// system's C++ runtime.

#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>

static void put(char *err, size_t len, const char *what) {
    if (len == 0) return;
    std::strncpy(err, what, len - 1);
    err[len - 1] = 0;
}

// Calls f(arg); 0 when it returned, 1 when it threw (its message in err).
extern "C" int torc_npu_guard(void (*f)(void *), void *arg, char *err, size_t len) {
    try {
        f(arg);
        return 0;
    } catch (const std::exception &e) {
        put(err, len, e.what());
    } catch (...) {
        put(err, len, "an exception that is not a std::exception");
    }
    return 1;
}

// For the test: throws, as ggml does when a buffer cannot be mapped.
extern "C" void torc_npu_guard_throw(void *) {
    throw std::runtime_error("buffer mapping failed (test)");
}
