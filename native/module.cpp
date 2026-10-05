#include <pybind11/pybind11.h>

void bind_camera(pybind11::module_&);
void bind_inference(pybind11::module_&);
void bind_features(pybind11::module_&);
void bind_movenet(pybind11::module_&);

PYBIND11_MODULE(_motion_native, m) {
    m.doc() = "meganan C++ CPU capture, inference, and movement engine";
    m.attr("engine_version") = "1";
    bind_camera(m);
    bind_inference(m);
    bind_features(m);
    bind_movenet(m);
}
