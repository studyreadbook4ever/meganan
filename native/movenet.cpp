// Native MoveNet inference through LiteRT's exported C API. The model and
// letterbox coordinates are identical to the Python reference backend.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "frame.hpp"
#include "resize.hpp"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#include <array>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace {

#define LITERT_FUNCTIONS(X) \
    X(LiteRtCreateEnvironment) X(LiteRtDestroyEnvironment) \
    X(LiteRtCreateModelFromFile) X(LiteRtDestroyModel) \
    X(LiteRtGetModelSignature) X(LiteRtGetNumSignatureInputs) \
    X(LiteRtGetNumSignatureOutputs) X(LiteRtGetSignatureInputTensorByIndex) \
    X(LiteRtGetSignatureOutputTensorByIndex) X(LiteRtGetRankedTensorType) \
    X(LiteRtCreateOptions) X(LiteRtDestroyOptions) \
    X(LiteRtSetOptionsHardwareAccelerators) \
    X(LiteRtCreateOpaqueOptions) X(LiteRtDestroyOpaqueOptions) X(LiteRtAddOpaqueOptions) \
    X(LiteRtCreateCompiledModel) X(LiteRtDestroyCompiledModel) \
    X(LiteRtGetCompiledModelInputBufferRequirements) \
    X(LiteRtGetCompiledModelOutputBufferRequirements) \
    X(LiteRtCreateManagedTensorBufferFromRequirements) \
    X(LiteRtDestroyTensorBuffer) X(LiteRtLockTensorBuffer) X(LiteRtUnlockTensorBuffer) \
    X(LiteRtRunCompiledModel)

class LiteRtLibrary {
public:
    explicit LiteRtLibrary(const std::string& path) {
        // LiteRT owns process-wide registries and worker TLS. Keep the DSO
        // resident after balancing dlopen/dlclose, while freeing every model
        // and tensor below, so late thread/TLS destructors retain valid code.
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (!handle_) throw std::runtime_error(std::string("Cannot load LiteRT: ") + dlerror());
        try {
#define LOAD(name) name = load<decltype(&::name)>(#name);
            LITERT_FUNCTIONS(LOAD)
#undef LOAD
        } catch (...) {
            dlclose(handle_);
            throw;
        }
    }
    ~LiteRtLibrary() { dlclose(handle_); }
#define DECLARE(name) decltype(&::name) name = nullptr;
    LITERT_FUNCTIONS(DECLARE)
#undef DECLARE
    void check(LiteRtStatus status, const char* operation) const {
        if (status != kLiteRtStatusOk) {
            throw std::runtime_error(std::string(operation) + ": LiteRT status " +
                                     std::to_string(static_cast<int>(status)));
        }
    }
private:
    void* handle_ = nullptr;
    template<class T> T load(const char* symbol) {
        dlerror();
        auto* function = dlsym(handle_, symbol);
        const auto* error = dlerror();
        if (error) throw std::runtime_error(std::string("Missing LiteRT C API ") + symbol + ": " + error);
        return reinterpret_cast<T>(function);
    }
};

class LockedBuffer {
public:
    LockedBuffer(LiteRtLibrary& library, LiteRtTensorBuffer buffer, LiteRtTensorBufferLockMode mode)
        : library_(library), buffer_(buffer) {
        library_.check(library_.LiteRtLockTensorBuffer(buffer_, &data, mode), "Lock inference buffer");
    }
    ~LockedBuffer() { library_.LiteRtUnlockTensorBuffer(buffer_); }
    void* data = nullptr;
private:
    LiteRtLibrary& library_;
    LiteRtTensorBuffer buffer_;
};

class NativeMoveNet {
public:
    NativeMoveNet(const std::string& library_path, const std::string& model_path)
        : library_(library_path) {
        try {
            library_.check(library_.LiteRtCreateEnvironment(0, nullptr, &environment_), "Create LiteRT environment");
            library_.check(library_.LiteRtCreateModelFromFile(environment_, model_path.c_str(), &model_), "Load MoveNet model");
            LiteRtSignature signature = nullptr;
            library_.check(library_.LiteRtGetModelSignature(model_, 0, &signature), "Get MoveNet signature");
            LiteRtParamIndex inputs = 0, outputs = 0;
            library_.check(library_.LiteRtGetNumSignatureInputs(signature, &inputs), "Count MoveNet inputs");
            library_.check(library_.LiteRtGetNumSignatureOutputs(signature, &outputs), "Count MoveNet outputs");
            if (inputs != 1 || outputs != 1) throw std::runtime_error("Unexpected MoveNet input/output count");
            LiteRtTensor input = nullptr, output = nullptr;
            library_.check(library_.LiteRtGetSignatureInputTensorByIndex(signature, 0, &input), "Get MoveNet input");
            library_.check(library_.LiteRtGetSignatureOutputTensorByIndex(signature, 0, &output), "Get MoveNet output");
            LiteRtRankedTensorType input_type{}, output_type{};
            library_.check(library_.LiteRtGetRankedTensorType(input, &input_type), "Inspect MoveNet input");
            library_.check(library_.LiteRtGetRankedTensorType(output, &output_type), "Inspect MoveNet output");
            validate_type(input_type, kLiteRtElementTypeUInt8, {1, 192, 192, 3});
            validate_type(output_type, kLiteRtElementTypeFloat32, {1, 1, 17, 3});

            library_.check(library_.LiteRtCreateOptions(&options_), "Create LiteRT options");
            library_.check(library_.LiteRtSetOptionsHardwareAccelerators(options_, kLiteRtHwAcceleratorCpu), "Select CPU inference");
            // v2.2.0's wheel exports opaque options but not the convenience
            // LrtCpuOptions setters. The pinned official runtime accepts this
            // TOML payload under its documented cpu_delegate identifier.
            LiteRtOpaqueOptions opaque = nullptr;
            // The C API requires a destructor callback even for borrowed
            // payloads; cpu_config_ outlives options_, so this one is a no-op.
            library_.check(library_.LiteRtCreateOpaqueOptions("cpu_delegate", cpu_config_.data(),
                                                            [](void*) {}, &opaque), "Configure CPU threads");
            const auto status = library_.LiteRtAddOpaqueOptions(options_, opaque);
            if (status != kLiteRtStatusOk) library_.LiteRtDestroyOpaqueOptions(opaque);
            library_.check(status, "Attach CPU options");
            library_.check(library_.LiteRtCreateCompiledModel(environment_, model_, options_, &compiled_), "Compile MoveNet for CPU");
            LiteRtTensorBufferRequirements input_requirements = nullptr, output_requirements = nullptr;
            library_.check(library_.LiteRtGetCompiledModelInputBufferRequirements(compiled_, 0, 0, &input_requirements), "Get MoveNet input buffer requirements");
            library_.check(library_.LiteRtGetCompiledModelOutputBufferRequirements(compiled_, 0, 0, &output_requirements), "Get MoveNet output buffer requirements");
            library_.check(library_.LiteRtCreateManagedTensorBufferFromRequirements(environment_, &input_type, input_requirements, &input_), "Allocate MoveNet input");
            library_.check(library_.LiteRtCreateManagedTensorBufferFromRequirements(environment_, &output_type, output_requirements, &output_), "Allocate MoveNet output");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~NativeMoveNet() { close(); }

    void close() {
        std::lock_guard<std::mutex> guard(mutex_);
        cleanup();
    }

    std::array<double, 51> infer(const std::uint8_t* data, int width, int height, bool bgr) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!compiled_) throw std::runtime_error("MoveNet has been closed");
        motion::Letterbox box{};
        {
            LockedBuffer input(library_, input_, kLiteRtTensorBufferLockModeWrite);
            box = motion::resize_movenet(data, width, height, bgr, static_cast<std::uint8_t*>(input.data));
        }
        library_.check(library_.LiteRtRunCompiledModel(compiled_, 0, 1, &input_, 1, &output_), "Run MoveNet");
        std::array<double, 51> result{};
        {
            LockedBuffer output(library_, output_, kLiteRtTensorBufferLockModeRead);
            const auto* points = static_cast<const float*>(output.data);
            for (int i = 0; i < 17; ++i) {
                result[i * 3] = (static_cast<double>(points[i * 3 + 1] * 192.f) - box.left) / box.scale / width;
                result[i * 3 + 1] = (static_cast<double>(points[i * 3] * 192.f) - box.top) / box.scale / height;
                result[i * 3 + 2] = points[i * 3 + 2];
            }
        }
        return result;
    }

private:
    LiteRtLibrary library_;
    std::mutex mutex_;
    std::string cpu_config_ = "num_threads = 2\nkernel_mode = \"delegate\"\n";
    LiteRtEnvironment environment_ = nullptr;
    LiteRtModel model_ = nullptr;
    LiteRtOptions options_ = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
    LiteRtTensorBuffer input_ = nullptr, output_ = nullptr;

    static void validate_type(const LiteRtRankedTensorType& type, LiteRtElementType element_type,
                              const std::array<int, 4>& shape) {
        if (type.element_type != element_type || type.layout.rank != 4 || type.layout.has_strides) {
            throw std::runtime_error("Unexpected MoveNet tensor format");
        }
        for (unsigned i = 0; i < 4; ++i) {
            if (type.layout.dimensions[i] != shape[i]) throw std::runtime_error("Unexpected MoveNet tensor shape");
        }
    }

    void cleanup() noexcept {
        if (input_) { library_.LiteRtDestroyTensorBuffer(input_); input_ = nullptr; }
        if (output_) { library_.LiteRtDestroyTensorBuffer(output_); output_ = nullptr; }
        if (compiled_) { library_.LiteRtDestroyCompiledModel(compiled_); compiled_ = nullptr; }
        if (options_) { library_.LiteRtDestroyOptions(options_); options_ = nullptr; }
        if (model_) { library_.LiteRtDestroyModel(model_); model_ = nullptr; }
        if (environment_) { library_.LiteRtDestroyEnvironment(environment_); environment_ = nullptr; }
    }
};

py::buffer_info checked_bgr(const py::array& array) {
    auto info = array.request();
    if (info.ndim != 3 || info.shape[2] != 3 || info.itemsize != 1 ||
        info.format != py::format_descriptor<std::uint8_t>::format() ||
        !(array.flags() & py::array::c_style) || info.shape[0] <= 0 || info.shape[1] <= 0 ||
        info.shape[0] > 16384 || info.shape[1] > 16384) {
        throw std::invalid_argument("Expected a contiguous HxWx3 uint8 BGR image");
    }
    return info;
}

py::array_t<double> result_array(const std::array<double, 51>& result) {
    py::array_t<double> array({17, 3});
    std::memcpy(array.mutable_data(), result.data(), sizeof(result));
    return array;
}

}  // namespace

void bind_movenet(py::module_& module) {
    py::class_<NativeMoveNet>(module, "NativeMoveNet")
        .def(py::init<const std::string&, const std::string&>(), py::arg("lib_path"), py::arg("model_path"),
             py::call_guard<py::gil_scoped_release>())
        .def("infer", [](NativeMoveNet& model, const std::shared_ptr<motion::Frame>& frame, py::object) {
            if (!frame || frame->width <= 0 || frame->height <= 0 ||
                frame->rgb.size() != static_cast<std::size_t>(frame->width) * frame->height * 3) {
                throw std::invalid_argument("Invalid native camera frame");
            }
            std::array<double, 51> result;
            {
                py::gil_scoped_release release;
                result = model.infer(frame->rgb.data(), frame->width, frame->height, false);
            }
            return result_array(result);
        }, py::arg("frame"), py::arg("timestamp") = py::none())
        .def("infer_bgr", [](NativeMoveNet& model, const py::array& frame, py::object) {
            const auto info = checked_bgr(frame);
            std::array<double, 51> result;
            {
                py::gil_scoped_release release;
                result = model.infer(static_cast<const std::uint8_t*>(info.ptr), static_cast<int>(info.shape[1]),
                                     static_cast<int>(info.shape[0]), true);
            }
            return result_array(result);
        }, py::arg("frame"), py::arg("timestamp") = py::none())
        .def_static("preprocess_bgr", [](const py::array& frame) {
            const auto info = checked_bgr(frame);
            py::array_t<std::uint8_t> tensor({1, 192, 192, 3});
            auto* output = tensor.mutable_data();
            {
                py::gil_scoped_release release;
                motion::resize_movenet(static_cast<const std::uint8_t*>(info.ptr), static_cast<int>(info.shape[1]),
                                      static_cast<int>(info.shape[0]), true, output);
            }
            return tensor;
        })
        .def("close", &NativeMoveNet::close, py::call_guard<py::gil_scoped_release>());
}
