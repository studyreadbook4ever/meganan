// Native temporal analysis and presentation for meganan.
// The Python modules remain the behavioral reference for this implementation.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Points = std::array<std::array<double, 3>, 17>;
constexpr const char* shape_error = "points must have shape (17, 3): x, y, confidence";

double timestamp_value(py::handle value, const std::optional<double>& previous) {
    py::object number = py::reinterpret_steal<py::object>(PyNumber_Float(value.ptr()));
    if (!number) throw py::error_already_set();
    const double timestamp = PyFloat_AsDouble(number.ptr());
    if (!std::isfinite(timestamp))
        throw py::value_error("timestamp must be finite monotonic seconds");
    if (previous && timestamp <= *previous)
        throw py::value_error("timestamps must increase strictly");
    return timestamp;
}

Array as_array(py::handle value) {
    // array_t's converter uses NumPy's conversion machinery at the boundary;
    // all geometry, filtering, and state transitions below run in C++.
    Array result(py::reinterpret_borrow<py::object>(value));
    // C-contiguous NumPy views may still start at an unaligned byte offset.
    // Copy bytes before dereferencing doubles; x86 tolerates these reads, but
    // they are undefined in C++ and can trap on other supported CPUs.
    if (reinterpret_cast<std::uintptr_t>(result.data()) % alignof(double)) {
        const std::vector<py::ssize_t> shape(result.shape(), result.shape() + result.ndim());
        Array aligned(shape);
        std::memcpy(aligned.mutable_data(), result.data(), result.nbytes());
        return aligned;
    }
    return result;
}

py::object optional_float(const std::optional<double>& value) {
    return value ? py::object(py::float_(*value)) : py::object(py::none());
}

double norm(double x, double y) { return std::sqrt(x * x + y * y); }

struct Debounced {
    std::string value = "UNKNOWN", candidate = "UNKNOWN";
    std::optional<double> since;
    bool update(const std::string& next, double timestamp, double delay) {
        if (next == "UNKNOWN") {
            value = candidate = "UNKNOWN";
            since.reset();
            return false;
        }
        if (next != candidate) {
            candidate = next;
            since = timestamp;
        }
        if (since && timestamp - *since + 1e-9 >= delay && value != next) {
            value = next;
            return true;
        }
        return false;
    }
};

class MotionAnalyzer {
public:
    double threshold, aspect_ratio, smoothing_seconds, debounce_seconds, loss_seconds;

    MotionAnalyzer(double confidence_threshold, double aspect, double smoothing,
                   double debounce, double loss)
        : threshold(confidence_threshold), aspect_ratio(aspect),
          smoothing_seconds(smoothing), debounce_seconds(debounce), loss_seconds(loss) {
        if (!(threshold > 0 && threshold <= 1))
            throw py::value_error("confidence_threshold must be in (0, 1]");
        if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0)
            throw py::value_error("aspect_ratio must be finite and positive");
        if (!std::isfinite(smoothing) || smoothing < 0 ||
            !std::isfinite(debounce) || debounce < 0 ||
            !std::isfinite(loss) || loss < 0)
            throw py::value_error("time constants must be finite and nonnegative");
    }

    py::dict update(py::object points, py::object time) {
        const double timestamp = timestamp_value(time, previous_time_);
        Points current{};
        if (!points.is_none()) {
            const auto source = as_array(points);
            if (source.ndim() != 2 || source.shape(0) != 17 || source.shape(1) != 3)
                throw py::value_error(shape_error);
            const double* data = source.data();
            for (int i = 0; i < 17; ++i) {
                const double x = data[3*i], y = data[3*i+1], c = data[3*i+2];
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(c) &&
                    x >= 0 && x <= 1 && y >= 0 && y <= 1 && c >= threshold && c <= 1)
                    current[i] = {x, y, c};
            }
        }
        std::array<bool, 17> valid{};
        for (int i = 0; i < 17; ++i) valid[i] = current[i][2] >= threshold;
        const double dt = previous_time_ ? timestamp - *previous_time_ : 0.;
        const bool consecutive = previous_time_.has_value() && dt <= .25;
        if (consecutive && smoothing_seconds > 0) {
            const double alpha = 1 - std::exp(-dt / smoothing_seconds);
            for (int i = 0; i < 17; ++i)
                if (valid[i] && previous_points_[i][2] >= threshold)
                    for (int j = 0; j < 2; ++j)
                        current[i][j] = alpha * current[i][j] +
                                        (1 - alpha) * previous_points_[i][j];
        }
        std::array<std::array<double, 2>, 17> metric{};
        for (int i = 0; i < 17; ++i) metric[i] = {current[i][0] * aspect_ratio, current[i][1]};
        const double width = norm(metric[5][0] - metric[6][0], metric[5][1] - metric[6][1]);
        const auto count = [&](int first, int end) {
            int result = 0;
            for (int i = first; i < end; ++i) result += valid[i];
            return result;
        };
        const bool tracked = valid[5] && valid[6] && count(7, 13) >= 2 && width >= .035;
        const bool partial = (count(0, 13) >= 3 && count(5, 7) && count(7, 11)) ||
                             (count(0, 5) >= 3 && count(5, 7)) ||
                             (valid[5] && valid[6] && width >= .035);
        const bool present = tracked || partial;
        const char* level = tracked ? (count(11, 17) == 6 ? "FULL" : "UPPER_BODY") :
                            present ? "PARTIAL" : "NONE";
        py::list events;
        const char* status;
        if (tracked) {
            if (!tracking_session_) {
                tracking_session_ = true;
                events.append("TRACKING_ACQUIRED");
            }
            last_seen_ = timestamp;
            status = "TRACKING";
        } else {
            const bool expired = !last_seen_ || timestamp - *last_seen_ >= loss_seconds;
            status = expired ? "SEARCHING" : "OCCLUDED";
            if (expired && tracking_session_) {
                events.append("TRACKING_LOST");
                tracking_session_ = false;
            }
            if (present) status = "PARTIAL";
        }
        py::dict angles;
        for (int side = 0; side < 2; ++side) {
            const int shoulder = 5 + side, elbow = 7 + side, wrist = 9 + side;
            auto& state = arms_[side];
            std::string candidate = "UNKNOWN";
            std::optional<double> angle;
            if (tracked && valid[shoulder] && valid[elbow] && valid[wrist]) {
                const double elevation = (current[shoulder][1] - current[wrist][1]) / width;
                if (elevation >= .15) candidate = "UP";
                else if (elevation <= -.25) candidate = "DOWN";
                else if (state.value == "UP" && elevation > .05) candidate = "UP";
                else if (state.value == "DOWN" && elevation < -.12) candidate = "DOWN";
                const double ax = metric[shoulder][0] - metric[elbow][0];
                const double ay = metric[shoulder][1] - metric[elbow][1];
                const double bx = metric[wrist][0] - metric[elbow][0];
                const double by = metric[wrist][1] - metric[elbow][1];
                const double denominator = norm(ax, ay) * norm(bx, by);
                if (denominator > 1e-8) {
                    const double cosine = std::clamp((ax * bx + ay * by) / denominator, -1., 1.);
                    angle = std::acos(cosine) * (180. / std::acos(-1.));
                }
            }
            angles[side ? "right_elbow" : "left_elbow"] = optional_float(angle);
            if (state.update(candidate, timestamp, debounce_seconds))
                events.append(std::string(side ? "RIGHT_ARM_" : "LEFT_ARM_") + state.value);
        }
        std::string lean_candidate = "UNKNOWN";
        if (tracked && valid[11] && valid[12]) {
            const double sy = (metric[5][1] + metric[6][1]) / 2;
            const double hy = (metric[11][1] + metric[12][1]) / 2;
            if (hy - sy >= .15 * width) {
                const double sx = (metric[5][0] + metric[6][0]) / 2;
                const double hx = (metric[11][0] + metric[12][0]) / 2;
                const double offset = (sx - hx) / width;
                if (offset < -.20 || (lean_.value == "LEFT" && offset < -.12)) lean_candidate = "LEFT";
                else if (offset > .20 || (lean_.value == "RIGHT" && offset > .12)) lean_candidate = "RIGHT";
                else lean_candidate = "CENTER";
            }
        }
        if (lean_.update(lean_candidate, timestamp, debounce_seconds))
            events.append("LEAN_" + lean_.value);
        std::optional<double> speed;
        if (tracked && previous_tracked_ && consecutive && dt >= .001) {
            double squared_sum = 0;
            int common = 0;
            for (int i = 5; i < 13; ++i) {
                if (!valid[i] || previous_points_[i][2] < threshold) continue;
                const double x = (current[i][0] - previous_points_[i][0]) * aspect_ratio;
                const double y = current[i][1] - previous_points_[i][1];
                squared_sum += x * x + y * y;
                ++common;
            }
            if (common >= 4) {
                const double raw = std::sqrt(squared_sum / common) / (width * dt);
                const double alpha = 1 - std::exp(-dt / .15);
                speed = speed_ ? alpha * raw + (1 - alpha) * *speed_ : raw;
            }
        }
        speed_ = speed;
        previous_points_ = current;
        previous_time_ = timestamp;
        previous_tracked_ = tracked;
        py::array_t<double> output_points({17, 3});
        std::memcpy(output_points.mutable_data(), current.data(), 51 * sizeof(double));
        py::dict result;
        result["points"] = std::move(output_points);
        result["tracked"] = tracked;
        result["person_present"] = present;
        result["tracking_level"] = level;
        result["status"] = status;
        result["left_arm"] = arms_[0].value;
        result["right_arm"] = arms_[1].value;
        result["lean"] = lean_.value;
        result["angles"] = std::move(angles);
        result["motion_speed"] = optional_float(speed);
        result["events"] = std::move(events);
        return result;
    }

private:
    Points previous_points_{};
    std::optional<double> previous_time_, last_seen_, speed_;
    bool previous_tracked_ = false, tracking_session_ = false;
    std::array<Debounced, 2> arms_;
    Debounced lean_;
};

// Keep ordinary frame dictionaries cheap while retaining deepcopy's semantics
// for unknown extension fields, ndarray subclasses, shared references and cycles.
py::object detach(py::handle source, py::dict& memo, const py::object& deepcopy) {
    if (source.is_none() || PyBool_Check(source.ptr()) || PyLong_CheckExact(source.ptr()) ||
        PyFloat_CheckExact(source.ptr()) || PyUnicode_CheckExact(source.ptr()) ||
        PyBytes_CheckExact(source.ptr()))
        return py::reinterpret_borrow<py::object>(source);
    const py::int_ identity(reinterpret_cast<std::uintptr_t>(source.ptr()));
    if (memo.contains(identity)) return py::reinterpret_borrow<py::object>(memo[identity]);
    if (PyDict_CheckExact(source.ptr())) {
        py::dict result;
        memo[identity] = result;
        for (const auto item : py::reinterpret_borrow<py::dict>(source))
            result[detach(item.first, memo, deepcopy)] = detach(item.second, memo, deepcopy);
        return result;
    }
    if (PyList_CheckExact(source.ptr())) {
        auto input = py::reinterpret_borrow<py::list>(source);
        py::list result(input.size());
        memo[identity] = result;
        for (py::size_t i = 0; i < input.size(); ++i)
            result[i] = detach(input[i], memo, deepcopy);
        return result;
    }
    return deepcopy(source, memo);
}

class MotionPresentation {
public:
    double threshold;
    explicit MotionPresentation(double confidence_threshold)
        : threshold(confidence_threshold), deepcopy_(py::module_::import("copy").attr("deepcopy")) {
        if (!std::isfinite(threshold) || !(threshold > 0 && threshold <= 1))
            throw py::value_error("confidence_threshold must be finite and in (0, 1]");
    }

    py::dict update(py::dict body, py::object time, py::object face) {
        const double timestamp = timestamp_value(time, previous_time_);
        py::dict memo;
        py::dict result = py::reinterpret_borrow<py::dict>(detach(body, memo, deepcopy_));
        Points empty{};
        Array points;
        const double* pose = reinterpret_cast<const double*>(empty.data());
        if (body.contains("points")) {
            points = as_array(body["points"]);
            if (points.ndim() != 2 || points.shape(0) != 17 || points.shape(1) != 3)
                throw py::value_error(shape_error);
            pose = points.data();
        }
        std::array<bool, 5> valid{};
        int valid_count = 0;
        for (int i = 0; i < 5; ++i) {
            const double x = pose[3*i], y = pose[3*i+1], c = pose[3*i+2];
            valid[i] = std::isfinite(x) && std::isfinite(y) && std::isfinite(c) &&
                       x >= 0 && x <= 1 && y >= 0 && y <= 1 && c >= threshold && c <= 1;
            valid_count += valid[i];
        }
        const bool eye_pair = valid[1] && valid[2], ear_pair = valid[3] && valid[4];
        const double eye_span = eye_pair ? norm(pose[3] - pose[6], pose[4] - pose[7]) : 0.;
        const double ear_span = ear_pair ? norm(pose[9] - pose[12], pose[10] - pose[13]) : 0.;
        const bool pose_tracked = valid_count >= 3 && std::max(eye_span, ear_span) >= .012;
        Array mesh;
        bool mesh_tracked = false;
        py::dict detection;
        if (py::isinstance<py::dict>(face)) {
            detection = py::reinterpret_borrow<py::dict>(face);
            if (detection.contains("detected") && py::cast<bool>(py::bool_(detection["detected"]))) {
                try {
                    mesh = as_array(detection.contains("landmarks") ?
                                    py::object(detection["landmarks"]) : py::object(py::none()));
                } catch (py::error_already_set& error) {
                    if (!error.matches(PyExc_ValueError) && !error.matches(PyExc_TypeError)) throw;
                    error.restore();
                    PyErr_Clear();
                } catch (const py::type_error&) {
                    // Numeric conversion failure has the same pose fallback as Python.
                }
                if (mesh && mesh.ndim() == 2 && mesh.shape(1) == 3 &&
                    (mesh.shape(0) == 468 || mesh.shape(0) == 478)) {
                    mesh_tracked = true;
                    const double* data = mesh.data();
                    for (py::ssize_t i = 0; i < mesh.size(); ++i)
                        if (!std::isfinite(data[i])) { mesh_tracked = false; break; }
                    mesh_tracked = mesh_tracked && data[3] >= 0 && data[3] <= 1 &&
                                                   data[4] >= 0 && data[4] <= 1;
                }
            }
        }
        const bool face_tracked = mesh_tracked || pose_tracked;
        const bool body_tracked = body.contains("tracked") && py::cast<bool>(py::bool_(body["tracked"]));
        result["body_tracked"] = body_tracked;
        result["face_tracked"] = face_tracked;
        result["face_mode"] = mesh_tracked ? "mesh" : pose_tracked ? "pose" : "none";
        result["tracked"] = body_tracked || face_tracked;
        const bool body_present = body.contains("person_present") ?
            py::cast<bool>(py::bool_(body["person_present"])) : body_tracked;
        result["person_present"] = body_present || face_tracked;
        if (face_tracked && !body_tracked) {
            result["tracking_level"] = "FACE";
            result["status"] = "FACE_TRACKING";
        }
        result["face_center"] = py::none();
        result["face_motion"] = py::none();
        result["face_bbox"] = py::none();
        py::list landmarks(mesh_tracked ? mesh.shape(0) : 0);
        if (mesh_tracked) {
            const double* data = mesh.data();
            for (py::ssize_t i = 0; i < mesh.shape(0); ++i) {
                py::list point(3);
                for (int j = 0; j < 3; ++j) point[j] = data[3*i+j];
                landmarks[i] = std::move(point);
            }
            // Missing/invalid numeric bbox uses the mesh envelope. Conversion
            // errors are intentionally raised, matching the reference adapter.
            auto supplied_bbox = as_array(detection.contains("bbox") ?
                                         py::object(detection["bbox"]) : py::object(py::none()));
            std::array<double, 4> bbox{};
            bool bbox_valid = supplied_bbox.ndim() == 1 && supplied_bbox.shape(0) == 4;
            if (bbox_valid) {
                std::copy_n(supplied_bbox.data(), 4, bbox.begin());
                for (double x : bbox) bbox_valid = bbox_valid && std::isfinite(x);
                bbox_valid = bbox_valid && bbox[0] <= bbox[2] && bbox[1] <= bbox[3];
            }
            if (!bbox_valid) {
                bbox = {data[0], data[1], data[0], data[1]};
                for (py::ssize_t i = 1; i < mesh.shape(0); ++i) {
                    bbox[0] = std::min(bbox[0], data[3*i]);
                    bbox[1] = std::min(bbox[1], data[3*i+1]);
                    bbox[2] = std::max(bbox[2], data[3*i]);
                    bbox[3] = std::max(bbox[3], data[3*i+1]);
                }
            }
            py::list output_bbox(4);
            for (int i = 0; i < 4; ++i) output_bbox[i] = std::clamp(bbox[i], 0., 1.);
            result["face_bbox"] = std::move(output_bbox);
        }
        result["face_landmarks"] = std::move(landmarks);
        py::list events;
        if (result.contains("events")) {
            PyObject* copied_events = PySequence_List(result["events"].ptr());
            if (!copied_events) throw py::error_already_set();
            events = py::reinterpret_steal<py::list>(copied_events);
        }
        if (face_tracked != previous_face_tracked_)
            events.append(face_tracked ? "FACE_ACQUIRED" : "FACE_LOST");
        result["events"] = std::move(events);
        std::string anchor;
        std::optional<std::array<double, 2>> center;
        if (face_tracked) {
            std::optional<double> confidence;
            if (mesh_tracked) {
                anchor = "mesh_nose";
                center = std::array<double, 2>{mesh.data()[3], mesh.data()[4]};
            } else {
                const int first = valid[0] ? 0 : eye_pair ? 1 : 3;
                const int second = valid[0] ? 0 : first + 1;
                anchor = valid[0] ? "nose" : eye_pair ? "eyes" : "ears";
                center = std::array<double, 2>{(pose[3*first] + pose[3*second]) / 2,
                                              (pose[3*first+1] + pose[3*second+1]) / 2};
                confidence = std::min(pose[3*first+2], pose[3*second+2]);
            }
            py::dict output_center;
            output_center["x"] = (*center)[0];
            output_center["y"] = (*center)[1];
            output_center["confidence"] = optional_float(confidence);
            result["face_center"] = std::move(output_center);
            const double dt = previous_time_ ? timestamp - *previous_time_ : 0;
            if (previous_face_tracked_ && anchor == previous_anchor_ && previous_center_ &&
                previous_time_ && dt >= .001 && dt <= .25) {
                const double dx = ((*center)[0] - (*previous_center_)[0]) / dt;
                const double dy = ((*center)[1] - (*previous_center_)[1]) / dt;
                py::dict movement;
                movement["dx"] = dx;
                movement["dy"] = dy;
                movement["speed"] = norm(dx, dy);
                movement["unit"] = "image_fraction_per_second";
                result["face_motion"] = std::move(movement);
            }
        }
        previous_time_ = timestamp;
        previous_face_tracked_ = face_tracked;
        previous_center_ = center;
        previous_anchor_ = anchor;
        return result;
    }

private:
    py::object deepcopy_;
    std::optional<double> previous_time_;
    std::optional<std::array<double, 2>> previous_center_;
    bool previous_face_tracked_ = false;
    std::string previous_anchor_;
};
} // namespace

void bind_features(py::module_& module) {
    py::class_<MotionAnalyzer>(module, "MotionAnalyzer")
        .def(py::init<double, double, double, double, double>(),
             py::arg("confidence_threshold") = .30, py::kw_only(),
             py::arg("aspect_ratio") = 1., py::arg("smoothing_seconds") = .07,
             py::arg("debounce_seconds") = .18, py::arg("loss_seconds") = .5)
        .def("update", &MotionAnalyzer::update, py::arg("points"), py::arg("timestamp"))
        .def_readwrite("threshold", &MotionAnalyzer::threshold)
        .def_readwrite("aspect_ratio", &MotionAnalyzer::aspect_ratio)
        .def_readwrite("smoothing_seconds", &MotionAnalyzer::smoothing_seconds)
        .def_readwrite("debounce_seconds", &MotionAnalyzer::debounce_seconds)
        .def_readwrite("loss_seconds", &MotionAnalyzer::loss_seconds);
    py::class_<MotionPresentation>(module, "MotionPresentation")
        .def(py::init<double>(), py::arg("confidence_threshold") = .3)
        .def("update", &MotionPresentation::update, py::arg("body_state"),
             py::arg("timestamp"), py::arg("face") = py::none())
        .def_readwrite("threshold", &MotionPresentation::threshold);
}
