#include <pybind11/pybind11.h>

#include "frame.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <jpeglib.h>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace py = pybind11;
namespace motion {
namespace {

using Clock = std::chrono::steady_clock;

double monotonic_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int camera_ioctl(int fd, unsigned long request, void* arg) {
    int result;
    do {
        result = ::ioctl(fd, request, arg);
    } while (result == -1 && errno == EINTR);
    return result;
}

[[noreturn]] void system_error(const std::string& operation) {
    const int error = errno;
    throw std::runtime_error(operation + ": " + std::strerror(error));
}

void disable_camera_core_dumps() {
    // A camera process must not persist RGB buffers through a crash dump.
    // RLIMIT_CORE alone is insufficient when Linux core_pattern pipes dumps
    // to a collector such as systemd-coredump; dumpability also must be off.
    const rlimit no_core{0, 0};
    if (::setrlimit(RLIMIT_CORE, &no_core) == -1) {
        system_error("Disable camera process core files");
    }
    if (::prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) == -1) {
        system_error("Disable camera process crash collection");
    }
}

struct Mapping {
    void* address = MAP_FAILED;
    std::size_t length = 0;
};

// A single capture thread owns all device operations and mapped memory.
// Cleanup also runs if any intermediate setup operation throws.
class CaptureSession {
public:
    int fd = -1;
    bool streaming = false;
    std::uint32_t format = 0;
    unsigned stride = 0;
    std::vector<Mapping> buffers;

    CaptureSession() = default;
    CaptureSession(const CaptureSession&) = delete;
    CaptureSession& operator=(const CaptureSession&) = delete;

    ~CaptureSession() {
        if (streaming) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            camera_ioctl(fd, VIDIOC_STREAMOFF, &type);
        }
        for (const auto& buffer : buffers) {
            if (buffer.address != MAP_FAILED) {
                ::munmap(buffer.address, buffer.length);
            }
        }
        if (fd != -1) ::close(fd);
    }

    void queue(unsigned index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (camera_ioctl(fd, VIDIOC_QBUF, &buffer) == -1) {
            system_error("Queue camera buffer");
        }
    }

    void open(const std::string& device) {
        fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd == -1) system_error("Cannot open " + device);

        v4l2_capability capability{};
        if (camera_ioctl(fd, VIDIOC_QUERYCAP, &capability) == -1) {
            system_error("Query camera capabilities for " + device);
        }
        const auto caps = (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                              ? capability.device_caps : capability.capabilities;
        if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING)) {
            throw std::runtime_error("Camera requires V4L2 capture and streaming: " + device);
        }

        // Prefer the camera's compressed USB stream. libjpeg-turbo decodes
        // directly to the RGB allocation consumed by the inference engines.
        bool configured = false;
        for (const auto pixel_format : {V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_YUYV}) {
            v4l2_format requested{};
            requested.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            requested.fmt.pix.width = 640;
            requested.fmt.pix.height = 480;
            requested.fmt.pix.pixelformat = pixel_format;
            requested.fmt.pix.field = V4L2_FIELD_ANY;
            if (camera_ioctl(fd, VIDIOC_S_FMT, &requested) == -1) continue;
            const auto& actual = requested.fmt.pix;
            if (actual.width != 640 || actual.height != 480) continue;
            if (actual.pixelformat != V4L2_PIX_FMT_MJPEG &&
                actual.pixelformat != V4L2_PIX_FMT_YUYV) continue;
            format = actual.pixelformat;
            stride = std::max(actual.bytesperline, 640u * 2u);
            configured = true;
            break;
        }
        if (!configured) {
            throw std::runtime_error("Camera does not provide 640x480 MJPEG or YUYV frames");
        }

        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator = 30;
        // Some cameras use a fixed frame rate and do not support S_PARM.
        if (camera_ioctl(fd, VIDIOC_S_PARM, &parameters) == -1 &&
            errno != EINVAL && errno != ENOTTY) {
            system_error("Set camera frame rate");
        }

        v4l2_requestbuffers request{};
        request.count = 4;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        if (camera_ioctl(fd, VIDIOC_REQBUFS, &request) == -1) {
            system_error("Allocate camera buffers");
        }
        if (request.count < 2 || request.count > 64) {
            throw std::runtime_error("Camera returned an unsupported buffer count");
        }
        buffers.resize(request.count);
        for (unsigned index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            buffer.type = request.type;
            buffer.memory = request.memory;
            buffer.index = index;
            if (camera_ioctl(fd, VIDIOC_QUERYBUF, &buffer) == -1) {
                system_error("Query camera buffer");
            }
            if (buffer.length == 0 || buffer.length > 32u * 1024u * 1024u) {
                throw std::runtime_error("Camera returned an invalid buffer size");
            }
            buffers[index].length = buffer.length;
            buffers[index].address = ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, fd, buffer.m.offset);
            if (buffers[index].address == MAP_FAILED) system_error("Map camera buffer");
            queue(index);
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (camera_ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
            system_error("Start camera stream");
        }
        streaming = true;
    }
};

struct JpegError {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
};

void jpeg_error_exit(j_common_ptr decoder) {
    auto* error = reinterpret_cast<JpegError*>(decoder->err);
    error->base.format_message(decoder, error->message);
    std::longjmp(error->jump, 1);
}

void jpeg_quiet_output(j_common_ptr) {}

void decode_jpeg(const std::uint8_t* source, std::size_t length, Frame& frame) {
    jpeg_decompress_struct decoder{};
    JpegError error{};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jpeg_error_exit;
    error.base.output_message = jpeg_quiet_output;
    // No C++ objects with destructors are created between setjmp and jpeg
    // calls: libjpeg's error callback must not jump across C++ lifetimes.
    volatile bool created = false;
    if (setjmp(error.jump)) {
        if (created) jpeg_destroy_decompress(&decoder);
        throw std::runtime_error(std::string("Invalid camera JPEG: ") + error.message);
    }
    jpeg_create_decompress(&decoder);
    created = true;
    jpeg_mem_src(&decoder, source, static_cast<unsigned long>(length));
    jpeg_read_header(&decoder, TRUE);
    if (decoder.image_width != static_cast<unsigned>(frame.width) ||
        decoder.image_height != static_cast<unsigned>(frame.height)) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error("Camera JPEG dimensions changed unexpectedly");
    }
    decoder.out_color_space = JCS_RGB;
    decoder.dct_method = JDCT_ISLOW;
    jpeg_start_decompress(&decoder);
    if (decoder.output_components != 3 ||
        decoder.output_width != static_cast<unsigned>(frame.width) ||
        decoder.output_height != static_cast<unsigned>(frame.height)) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error("Unsupported camera JPEG output format");
    }
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = frame.rgb.data() + decoder.output_scanline * frame.width * 3;
        jpeg_read_scanlines(&decoder, &row, 1);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
}

std::uint8_t clamp_byte(int value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

void decode_yuyv(const std::uint8_t* source, std::size_t length, unsigned stride,
                 Frame& frame) {
    const auto required = static_cast<std::size_t>(stride) * (frame.height - 1) +
                          static_cast<std::size_t>(frame.width) * 2;
    if (length < required) throw std::runtime_error("Truncated camera YUYV frame");
    for (int y = 0; y < frame.height; ++y) {
        const auto* input = source + static_cast<std::size_t>(y) * stride;
        auto* output = frame.rgb.data() + static_cast<std::size_t>(y) * frame.width * 3;
        for (int x = 0; x < frame.width; x += 2, input += 4, output += 6) {
            const int u = static_cast<int>(input[1]) - 128;
            const int v = static_cast<int>(input[3]) - 128;
            for (int k = 0; k < 2; ++k) {
                const int luma = 298 * std::max(0, static_cast<int>(input[k * 2]) - 16);
                output[k * 3] = clamp_byte((luma + 409 * v + 128) >> 8);
                output[k * 3 + 1] = clamp_byte((luma - 100 * u - 208 * v + 128) >> 8);
                output[k * 3 + 2] = clamp_byte((luma + 516 * u + 128) >> 8);
            }
        }
    }
}

struct Sample {
    std::uint64_t sequence = 0;
    double timestamp = 0;
    std::shared_ptr<Frame> frame;
    double fps = 0;
};

}  // namespace

class NativeCamera {
public:
    explicit NativeCamera(std::string device) : device_(std::move(device)) {
        if (device_.empty() || device_.find('\0') != std::string::npos) {
            throw std::invalid_argument("Camera device must be a nonempty path without NUL bytes");
        }
    }

    NativeCamera(const NativeCamera&) = delete;
    NativeCamera& operator=(const NativeCamera&) = delete;
    ~NativeCamera() { close(); }

    NativeCamera& start() {
        {
            std::lock_guard<std::mutex> control(control_mutex_);
            std::lock_guard<std::mutex> state(state_mutex_);
            if (stop_.load()) throw std::runtime_error("Camera has been closed");
            if (started_) throw std::runtime_error("Camera has already been started");
            disable_camera_core_dumps();
            started_ = true;
            try {
                worker_ = std::thread(&NativeCamera::run, this);
            } catch (...) {
                started_ = false;
                throw;
            }
        }
        std::unique_lock<std::mutex> state(state_mutex_);
        condition_.wait(state, [this] { return initialized_ || !error_.empty() || stop_.load(); });
        if (!error_.empty()) throw std::runtime_error(error_);
        if (stop_.load()) throw std::runtime_error("Camera has been closed");
        return *this;
    }

    Sample next(std::uint64_t after, double timeout) {
        // Bound before duration conversion: enormous finite floating-point
        // values can overflow steady_clock's signed nanosecond representation.
        if (!std::isfinite(timeout) || timeout < 0 || timeout > 3600) {
            throw std::invalid_argument("Camera timeout must be finite and between 0 and 3600 seconds");
        }
        std::unique_lock<std::mutex> state(state_mutex_);
        if (!started_) {
            if (stop_.load()) return {};
            throw std::runtime_error("Camera has not been started");
        }
        condition_.wait_for(state, std::chrono::duration<double>(timeout), [this, after] {
            return sequence_ > after || !error_.empty() || stop_.load();
        });
        if (!error_.empty()) throw std::runtime_error(error_);
        if (stop_.load() || sequence_ <= after) return {};
        double fps = 0;
        if (timestamps_.size() > 1 && timestamps_.back() > timestamps_.front()) {
            fps = (timestamps_.size() - 1) / (timestamps_.back() - timestamps_.front());
        }
        return {sequence_, timestamp_, latest_, fps};
    }

    void close() noexcept {
        std::lock_guard<std::mutex> control(control_mutex_);
        stop_.store(true);
        condition_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> state(state_mutex_);
        latest_.reset();
    }

    const std::string& device() const { return device_; }

private:
    std::string device_;
    std::mutex control_mutex_;
    std::mutex state_mutex_;
    std::condition_variable condition_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    bool started_ = false;
    bool initialized_ = false;
    std::string error_;
    std::uint64_t sequence_ = 0;
    double timestamp_ = 0;
    std::shared_ptr<Frame> latest_;
    std::deque<double> timestamps_;

    void run() noexcept {
        try {
            CaptureSession session;
            session.open(device_);
            {
                std::lock_guard<std::mutex> state(state_mutex_);
                initialized_ = true;
                condition_.notify_all();
            }
            auto last_success = Clock::now();
            unsigned bad_frames = 0;
            while (!stop_.load()) {
                pollfd descriptor{session.fd, POLLIN, 0};
                const int ready = ::poll(&descriptor, 1, 100);
                if (stop_.load()) break;
                if (ready < 0) {
                    if (errno == EINTR) continue;
                    system_error("Poll camera");
                }
                if (Clock::now() - last_success > std::chrono::seconds(5)) {
                    throw std::runtime_error("Camera stopped returning valid frames for 5 seconds");
                }
                if (ready == 0) continue;
                if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                    throw std::runtime_error("Camera stream disconnected or failed");
                }
                if (!(descriptor.revents & POLLIN)) continue;

                // Discard queued older buffers without decoding them. Retain
                // exactly the newest immutable frame, never an image backlog.
                v4l2_buffer newest{};
                bool have_buffer = false;
                for (std::size_t n = 0; n < session.buffers.size(); ++n) {
                    v4l2_buffer buffer{};
                    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                    buffer.memory = V4L2_MEMORY_MMAP;
                    if (camera_ioctl(session.fd, VIDIOC_DQBUF, &buffer) == -1) {
                        if (errno == EAGAIN || errno == EIO) break;
                        system_error("Read camera frame");
                    }
                    if (buffer.index >= session.buffers.size()) {
                        throw std::runtime_error("Camera returned an invalid buffer index");
                    }
                    if (have_buffer) session.queue(newest.index);
                    newest = buffer;
                    have_buffer = true;
                }
                if (!have_buffer) continue;
                const auto& mapped = session.buffers[newest.index];
                if ((newest.flags & V4L2_BUF_FLAG_ERROR) || newest.bytesused == 0 ||
                    newest.bytesused > mapped.length) {
                    session.queue(newest.index);
                    if (++bad_frames >= 20) throw std::runtime_error("Camera repeatedly returned invalid frames");
                    continue;
                }
                auto frame = std::make_shared<Frame>();
                frame->width = 640;
                frame->height = 480;
                frame->rgb.resize(640u * 480u * 3u);
                try {
                    const auto* source = static_cast<const std::uint8_t*>(mapped.address);
                    if (session.format == V4L2_PIX_FMT_MJPEG) {
                        decode_jpeg(source, newest.bytesused, *frame);
                    } else {
                        decode_yuyv(source, newest.bytesused, session.stride, *frame);
                    }
                } catch (const std::exception&) {
                    session.queue(newest.index);
                    if (++bad_frames >= 20) throw;
                    continue;
                }
                session.queue(newest.index);
                bad_frames = 0;
                last_success = Clock::now();
                const double now = monotonic_seconds();
                {
                    std::lock_guard<std::mutex> state(state_mutex_);
                    latest_ = std::move(frame);
                    timestamp_ = now;
                    ++sequence_;
                    timestamps_.push_back(now);
                    if (timestamps_.size() > 90) timestamps_.pop_front();
                    condition_.notify_all();
                }
            }
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> state(state_mutex_);
            if (!stop_.load()) error_ = error.what();
            condition_.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> state(state_mutex_);
            if (!stop_.load()) error_ = "Unexpected native camera failure";
            condition_.notify_all();
        }
    }
};

}  // namespace motion

void bind_camera(py::module_& module) {
    py::class_<motion::Frame, std::shared_ptr<motion::Frame>>(module, "CameraFrame")
        .def_property_readonly("width", [](const motion::Frame& frame) { return frame.width; })
        .def_property_readonly("height", [](const motion::Frame& frame) { return frame.height; });

    py::class_<motion::NativeCamera>(module, "NativeCamera")
        .def(py::init<std::string>(), py::arg("device") = "/dev/video0")
        .def_property_readonly("device", &motion::NativeCamera::device)
        .def("start", &motion::NativeCamera::start, py::return_value_policy::reference_internal,
             py::call_guard<py::gil_scoped_release>())
        .def("next", [](motion::NativeCamera& camera, std::uint64_t after, double timeout) -> py::object {
            motion::Sample sample;
            {
                py::gil_scoped_release release;
                sample = camera.next(after, timeout);
            }
            if (!sample.frame) return py::none();
            return py::make_tuple(sample.sequence, sample.timestamp, sample.frame, sample.fps);
        }, py::arg("after"), py::arg("timeout") = 0.3)
        .def("close", &motion::NativeCamera::close, py::call_guard<py::gil_scoped_release>());
}
