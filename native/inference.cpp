// meganan's native CPU inference bridge. Only numeric geometry is returned
// to Python. MediaPipe remains the same native implementation/model as before;
// the per-frame ctypes dispatcher and Python landmark objects are bypassed.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "frame.hpp"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace {

// All function types come directly from the unmodified, pinned official C
// headers in third_party/mediapipe. No inferred or duplicated ABI structs.
class MediaPipeLibrary {
 public:
  explicit MediaPipeLibrary(const std::string& path) {
    // MediaPipe owns process-wide registries and thread-local cleanup hooks.
    // Its Python binding also keeps the DSO loaded for the process lifetime.
    // Graph/image/result resources are still closed normally, but unmapping
    // code when a worker leaves can crash in a remaining TLS destructor.
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!handle_) throw std::runtime_error(std::string("Cannot load MediaPipe: ") + dlerror());
    try {
#define MP_LOAD(name) name = load<decltype(&::name)>(#name)
      MP_LOAD(MpErrorFree);
      MP_LOAD(MpImageCreateFromUint8Data);
      MP_LOAD(MpImageFree);
      MP_LOAD(MpPoseLandmarkerCreate);
      MP_LOAD(MpPoseLandmarkerDetectForVideo);
      MP_LOAD(MpPoseLandmarkerCloseResult);
      MP_LOAD(MpPoseLandmarkerClose);
      MP_LOAD(MpFaceLandmarkerCreate);
      MP_LOAD(MpFaceLandmarkerDetectForVideo);
      MP_LOAD(MpFaceLandmarkerCloseResult);
      MP_LOAD(MpFaceLandmarkerClose);
#undef MP_LOAD
    } catch (...) {
      dlclose(handle_);
      throw;
    }
  }
  ~MediaPipeLibrary() { if (!retain_loaded_) dlclose(handle_); }
  MediaPipeLibrary(const MediaPipeLibrary&) = delete;
  MediaPipeLibrary& operator=(const MediaPipeLibrary&) = delete;

  // A failing native Close can leave a graph alive. Its code must not be
  // unloaded during best-effort destruction; explicit close reports the error
  // and retains its task handle so callers can retry before this last resort.
  void retain_loaded() noexcept { retain_loaded_ = true; }

#define MP_SYMBOL(name) decltype(&::name) name = nullptr
  MP_SYMBOL(MpErrorFree);
  MP_SYMBOL(MpImageCreateFromUint8Data);
  MP_SYMBOL(MpImageFree);
  MP_SYMBOL(MpPoseLandmarkerCreate);
  MP_SYMBOL(MpPoseLandmarkerDetectForVideo);
  MP_SYMBOL(MpPoseLandmarkerCloseResult);
  MP_SYMBOL(MpPoseLandmarkerClose);
  MP_SYMBOL(MpFaceLandmarkerCreate);
  MP_SYMBOL(MpFaceLandmarkerDetectForVideo);
  MP_SYMBOL(MpFaceLandmarkerCloseResult);
  MP_SYMBOL(MpFaceLandmarkerClose);
#undef MP_SYMBOL

  void check(MpStatus status, char* error, const char* operation) const {
    // MediaPipe owns the allocation; MpErrorFree is its public deallocator.
    std::unique_ptr<char, decltype(MpErrorFree)> owned(error, MpErrorFree);
    if (status == 0) return;
    std::string message = std::string(operation) + ": " +
                          (error ? error : "MediaPipe operation failed");
    if (status == 3) throw std::invalid_argument(message);
    throw std::runtime_error(message);
  }

 private:
  template <class Function> Function load(const char* name) {
    dlerror();
    void* symbol = dlsym(handle_, name);
    const char* error = dlerror();
    if (error) throw std::runtime_error(std::string("Missing MediaPipe C API ") + name + ": " + error);
    return reinterpret_cast<Function>(symbol);
  }
  void* handle_ = nullptr;
  bool retain_loaded_ = false;
};

struct ImageResource {
  explicit ImageResource(MediaPipeLibrary& library) : lib(library) {}
  ~ImageResource() { if (value) lib.MpImageFree(value); }
  MediaPipeLibrary& lib;
  MpImagePtr value = nullptr;
};
struct PoseResultResource {
  explicit PoseResultResource(MediaPipeLibrary& library) : lib(library) {}
  ~PoseResultResource() { lib.MpPoseLandmarkerCloseResult(&value); }
  MediaPipeLibrary& lib;
  MpPoseLandmarkerResult value{};
};
struct FaceResultResource {
  explicit FaceResultResource(MediaPipeLibrary& library) : lib(library) {}
  ~FaceResultResource() { lib.MpFaceLandmarkerCloseResult(&value); }
  MediaPipeLibrary& lib;
  MpFaceLandmarkerResult value{};
};

struct Geometry {
  std::array<float, 51> pose{};
  std::vector<float> face;
  std::array<float, 4> bbox{};
};

constexpr std::array<unsigned, 17> coco_indices = {
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28};

void extract_pose(const MpPoseLandmarkerResult& result, Geometry& output) {
  if (!result.pose_landmarks_count || !result.pose_landmarks) return;
  const auto& landmarks = result.pose_landmarks[0];
  if (!landmarks.landmarks || landmarks.landmarks_count < 29) return;
  for (std::size_t i = 0; i != coco_indices.size(); ++i) {
    const auto& point = landmarks.landmarks[coco_indices[i]];
    float confidence = std::min(point.has_visibility ? point.visibility : 0.0f,
                                point.has_presence ? point.presence : 0.0f);
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(confidence)) continue;
    if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) confidence = 0;
    output.pose[i * 3] = point.x;
    output.pose[i * 3 + 1] = point.y;
    output.pose[i * 3 + 2] = std::clamp(confidence, 0.0f, 1.0f);
  }
}

void extract_face(const MpFaceLandmarkerResult& result, Geometry& output) {
  if (!result.face_landmarks_count || !result.face_landmarks) return;
  const auto& landmarks = result.face_landmarks[0];
  if (!landmarks.landmarks || landmarks.landmarks_count != 478) return;
  float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
  output.face.resize(478 * 3);
  for (std::size_t i = 0; i != 478; ++i) {
    const auto& point = landmarks.landmarks[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      output.face.clear();
      return;
    }
    output.face[i * 3] = point.x;
    output.face[i * 3 + 1] = point.y;
    output.face[i * 3 + 2] = point.z;
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  output.bbox = {std::clamp(min_x, 0.0f, 1.0f), std::clamp(min_y, 0.0f, 1.0f),
                 std::clamp(max_x, 0.0f, 1.0f), std::clamp(max_y, 0.0f, 1.0f)};
  if (output.bbox[2] <= output.bbox[0] || output.bbox[3] <= output.bbox[1]) output.face.clear();
}

py::tuple python_geometry(const Geometry& geometry) {
  py::array_t<float> pose({17, 3});
  std::memcpy(pose.mutable_data(), geometry.pose.data(), sizeof(geometry.pose));
  const bool detected = !geometry.face.empty();
  py::array_t<float> points({detected ? 478 : 0, 3});
  if (detected) std::memcpy(points.mutable_data(), geometry.face.data(), geometry.face.size() * sizeof(float));
  py::dict face;
  face["detected"] = detected;
  face["landmarks"] = std::move(points);
  face["contours"] = py::module_::import("motion_topology").attr("FACE_CONTOURS");
  if (detected) {
    py::list box;
    for (float value : geometry.bbox) box.append(value);
    face["bbox"] = std::move(box);
  } else {
    face["bbox"] = py::none();
  }
  return py::make_tuple(std::move(pose), std::move(face));
}

class NativeLandmarkers {
 public:
  NativeLandmarkers(const std::string& lib_path, const std::string& pose_path,
                    const std::string& face_path) : lib_(lib_path) {
    if (pose_path.empty() && face_path.empty()) throw std::invalid_argument("At least one landmark model is required");
    try {
      if (!pose_path.empty()) {
        validate_model(pose_path);
        MpPoseLandmarkerOptions options{};
        set_base_options(options.base_options, pose_path);
        options.running_mode = MP_RUNNING_MODE_VIDEO;
        options.num_poses = 1;
        options.min_pose_detection_confidence = 0.5f;
        options.min_pose_presence_confidence = 0.5f;
        options.min_tracking_confidence = 0.5f;
        options.output_segmentation_masks = false;
        options.result_callback = nullptr;
        char* error = nullptr;
        const auto status = lib_.MpPoseLandmarkerCreate(&options, &pose_, &error);
        lib_.check(status, error, "Create pose landmarker");
      }
      if (!face_path.empty()) {
        validate_model(face_path);
        MpFaceLandmarkerOptions options{};
        set_base_options(options.base_options, face_path);
        options.running_mode = MP_RUNNING_MODE_VIDEO;
        options.num_faces = 1;
        options.min_face_detection_confidence = 0.5f;
        options.min_face_presence_confidence = 0.5f;
        options.min_tracking_confidence = 0.5f;
        options.output_face_blendshapes = false;
        options.output_facial_transformation_matrixes = false;
        options.result_callback = nullptr;
        char* error = nullptr;
        const auto status = lib_.MpFaceLandmarkerCreate(&options, &face_, &error);
        lib_.check(status, error, "Create face landmarker");
      }
    } catch (...) {
      close_noexcept();
      throw;
    }
  }

  ~NativeLandmarkers() {
    // Inference/close can run on different Python threads. Never wait for the
    // native mutex while retaining the GIL: another caller may need it to exit.
    close_noexcept();
  }

  Geometry infer(const motion::Frame& frame, double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) throw std::runtime_error("Native landmark engine is closed");
    const auto bytes = validate_dimensions(frame.width, frame.height);
    if (frame.rgb.size() != bytes) throw std::invalid_argument("Invalid RGB frame buffer size");
    if (!std::isfinite(seconds) || seconds < 0 ||
        seconds >= static_cast<double>(std::numeric_limits<std::int64_t>::max() / 1000000 - 1)) {
      throw std::invalid_argument("timestamp must be finite, nonnegative monotonic seconds within int64 range");
    }
    // Match both old adapters: duplicate/backward timestamps are advanced by
    // one millisecond, so the MediaPipe VIDEO graph always sees increasing time.
    const auto timestamp_ms = std::max(static_cast<std::int64_t>(seconds * 1000), last_timestamp_ms_ + 1);
    ImageResource image(lib_);
    char* error = nullptr;
    auto status = lib_.MpImageCreateFromUint8Data(kMpImageFormatSrgb, frame.width,
        frame.height, frame.rgb.data(), static_cast<int>(bytes), &image.value, &error);
    lib_.check(status, error, "Create RGB image");
    Geometry geometry;
    // Reserve the timestamp even on an error: a task may already have consumed
    // it when another task fails. A retry must never reuse a consumed timestamp.
    last_timestamp_ms_ = timestamp_ms;
    if (pose_) {
      PoseResultResource result(lib_);
      error = nullptr;
      status = lib_.MpPoseLandmarkerDetectForVideo(pose_, image.value, nullptr,
          timestamp_ms, &result.value, &error);
      lib_.check(status, error, "Pose inference");
      extract_pose(result.value, geometry);
    }
    if (face_) {
      FaceResultResource result(lib_);
      error = nullptr;
      status = lib_.MpFaceLandmarkerDetectForVideo(face_, image.value, nullptr,
          timestamp_ms, &result.value, &error);
      lib_.check(status, error, "Face inference");
      extract_face(result.value, geometry);
    }
    return geometry;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    close_unlocked();
  }

  static std::size_t validate_dimensions(std::int64_t width, std::int64_t height) {
    if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() ||
        height > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max() / 3 / height) {
      throw std::invalid_argument("Expected a non-empty RGB/BGR image within supported dimensions");
    }
    return static_cast<std::size_t>(width * height * 3);
  }

 private:
  static void validate_model(const std::string& path) {
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Model file not found: " + path);
  }

  static void set_base_options(MpBaseOptions& options, const std::string& path) {
    options.model_asset_path = path.c_str();
    options.file_descriptor = -1;
    options.delegate = MP_DELEGATE_CPU;
    options.host_environment = MP_HOST_ENVIRONMENT_UNKNOWN;
    options.host_system = MP_HOST_SYSTEM_LINUX;
  }

  void close_unlocked() {
    closed_ = true;
    std::string failure;
    const auto close_task = [&](auto& handle, auto close_fn, const char* name) {
      if (!handle) return;
      char* error = nullptr;
      const auto status = close_fn(handle, &error);
      std::unique_ptr<char, decltype(lib_.MpErrorFree)> owned(error, lib_.MpErrorFree);
      if (status == 0) handle = nullptr;
      else if (failure.empty()) failure = std::string(name) + ": " +
          (error ? error : "MediaPipe close failed");
    };
    // Attempt both resources even when the first one reports failure.
    close_task(face_, lib_.MpFaceLandmarkerClose, "Close face landmarker");
    close_task(pose_, lib_.MpPoseLandmarkerClose, "Close pose landmarker");
    if (!failure.empty()) throw std::runtime_error(failure);
  }

  void close_noexcept() noexcept {
    try { close_unlocked(); } catch (...) {}
    if (pose_ || face_) lib_.retain_loaded();
  }

  MediaPipeLibrary lib_;
  MpPoseLandmarkerPtr pose_ = nullptr;
  MpFaceLandmarkerPtr face_ = nullptr;
  std::mutex mutex_;
  std::int64_t last_timestamp_ms_ = -1;
  bool closed_ = false;
};

// The ABI smoke test compares every used struct's size and each field offset
// against the installed SDK's ctypes declarations before invoking real models.
py::dict mediapipe_abi_layout() {
  py::dict result;
  py::dict item;
#define MP_STRUCT(type) item = py::dict(); item["size"] = sizeof(type); result[#type] = item
#define MP_FIELD(type, field) item[#field] = offsetof(type, field)
  MP_STRUCT(MpBaseOptions);
  MP_FIELD(MpBaseOptions, model_asset_buffer); MP_FIELD(MpBaseOptions, model_asset_buffer_count);
  MP_FIELD(MpBaseOptions, model_asset_path); MP_FIELD(MpBaseOptions, file_descriptor);
  MP_FIELD(MpBaseOptions, delegate); MP_FIELD(MpBaseOptions, host_environment);
  MP_FIELD(MpBaseOptions, host_system); MP_FIELD(MpBaseOptions, host_version);
  MP_FIELD(MpBaseOptions, ca_bundle_path); MP_FIELD(MpBaseOptions, app_id);
  MP_FIELD(MpBaseOptions, app_version);
  MP_STRUCT(MpNormalizedLandmark);
  MP_FIELD(MpNormalizedLandmark, x); MP_FIELD(MpNormalizedLandmark, y); MP_FIELD(MpNormalizedLandmark, z);
  MP_FIELD(MpNormalizedLandmark, has_visibility); MP_FIELD(MpNormalizedLandmark, visibility);
  MP_FIELD(MpNormalizedLandmark, has_presence); MP_FIELD(MpNormalizedLandmark, presence);
  MP_FIELD(MpNormalizedLandmark, name);
  MP_STRUCT(MpNormalizedLandmarks);
  MP_FIELD(MpNormalizedLandmarks, landmarks); MP_FIELD(MpNormalizedLandmarks, landmarks_count);
  MP_STRUCT(MpPoseLandmarkerOptions);
  MP_FIELD(MpPoseLandmarkerOptions, base_options); MP_FIELD(MpPoseLandmarkerOptions, running_mode);
  MP_FIELD(MpPoseLandmarkerOptions, num_poses); MP_FIELD(MpPoseLandmarkerOptions, min_pose_detection_confidence);
  MP_FIELD(MpPoseLandmarkerOptions, min_pose_presence_confidence); MP_FIELD(MpPoseLandmarkerOptions, min_tracking_confidence);
  MP_FIELD(MpPoseLandmarkerOptions, output_segmentation_masks); MP_FIELD(MpPoseLandmarkerOptions, result_callback);
  MP_STRUCT(MpFaceLandmarkerOptions);
  MP_FIELD(MpFaceLandmarkerOptions, base_options); MP_FIELD(MpFaceLandmarkerOptions, running_mode);
  MP_FIELD(MpFaceLandmarkerOptions, num_faces); MP_FIELD(MpFaceLandmarkerOptions, min_face_detection_confidence);
  MP_FIELD(MpFaceLandmarkerOptions, min_face_presence_confidence); MP_FIELD(MpFaceLandmarkerOptions, min_tracking_confidence);
  MP_FIELD(MpFaceLandmarkerOptions, output_face_blendshapes); MP_FIELD(MpFaceLandmarkerOptions, output_facial_transformation_matrixes);
  MP_FIELD(MpFaceLandmarkerOptions, result_callback);
  MP_STRUCT(MpPoseLandmarkerResult);
  MP_FIELD(MpPoseLandmarkerResult, segmentation_masks); MP_FIELD(MpPoseLandmarkerResult, segmentation_masks_count);
  MP_FIELD(MpPoseLandmarkerResult, pose_landmarks); MP_FIELD(MpPoseLandmarkerResult, pose_landmarks_count);
  MP_FIELD(MpPoseLandmarkerResult, pose_world_landmarks); MP_FIELD(MpPoseLandmarkerResult, pose_world_landmarks_count);
  MP_STRUCT(MpFaceLandmarkerResult);
  MP_FIELD(MpFaceLandmarkerResult, face_landmarks); MP_FIELD(MpFaceLandmarkerResult, face_landmarks_count);
  MP_FIELD(MpFaceLandmarkerResult, face_blendshapes); MP_FIELD(MpFaceLandmarkerResult, face_blendshapes_count);
  MP_FIELD(MpFaceLandmarkerResult, facial_transformation_matrixes); MP_FIELD(MpFaceLandmarkerResult, facial_transformation_matrixes_count);
#undef MP_STRUCT
#undef MP_FIELD
  return result;
}
}  // namespace

void bind_inference(py::module_& module) {
  module.def("mediapipe_abi_layout", &mediapipe_abi_layout);
  py::class_<NativeLandmarkers>(module, "NativeLandmarkers")
      .def(py::init([](const std::string& lib_path, const std::string& pose_model_path,
                       const std::string& face_model_path) {
        py::gil_scoped_release release;
        return std::make_unique<NativeLandmarkers>(lib_path, pose_model_path, face_model_path);
      }), py::arg("lib_path"), py::arg("pose_model_path") = "", py::arg("face_model_path") = "")
      .def("infer", [](NativeLandmarkers& engine, const std::shared_ptr<motion::Frame>& frame, double timestamp) {
        if (!frame) throw std::invalid_argument("Expected a native RGB frame");
        Geometry geometry;
        {
          py::gil_scoped_release release;
          geometry = engine.infer(*frame, timestamp);
        }
        return python_geometry(geometry);
      }, py::arg("frame"), py::arg("timestamp"))
      .def("infer_bgr", [](NativeLandmarkers& engine, py::object input, double timestamp) {
        if (!py::isinstance<py::array>(input)) throw std::invalid_argument("Expected a BGR image with shape (height, width, 3)");
        auto array = py::reinterpret_borrow<py::array>(input);
        const auto info = array.request();
        if (info.ndim != 3 || info.shape[2] != 3 || !array.dtype().is(py::dtype::of<std::uint8_t>())) {
          throw std::invalid_argument("Expected a uint8 BGR image with shape (height, width, 3)");
        }
        const auto bytes = NativeLandmarkers::validate_dimensions(info.shape[1], info.shape[0]);
        motion::Frame frame;
        frame.width = static_cast<int>(info.shape[1]);
        frame.height = static_cast<int>(info.shape[0]);
        Geometry geometry;
        {
          py::gil_scoped_release release;
          frame.rgb.resize(bytes);
          const auto* source = static_cast<const std::uint8_t*>(info.ptr);
          for (int y = 0; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
              const auto* pixel = source + y * info.strides[0] + x * info.strides[1];
              auto* target = frame.rgb.data() + (static_cast<std::size_t>(y) * frame.width + x) * 3;
              target[0] = pixel[2 * info.strides[2]];
              target[1] = pixel[info.strides[2]];
              target[2] = pixel[0];
            }
          }
          geometry = engine.infer(frame, timestamp);
        }
        return python_geometry(geometry);
      }, py::arg("frame"), py::arg("timestamp"))
      .def("close", &NativeLandmarkers::close, py::call_guard<py::gil_scoped_release>());
}
