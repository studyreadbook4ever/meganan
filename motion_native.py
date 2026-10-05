"""Thin configuration boundary for the C++ capture and movement engine.

Pixels stay in opaque C++ CameraFrame objects. Only numerical landmarks cross
back into Python for the existing GUI, API, and aggregate telemetry.
"""
from importlib import metadata, util
from pathlib import Path

try:
    from _motion_native import MotionAnalyzer, MotionPresentation, NativeCamera, NativeLandmarkers
except ImportError as exc:
    raise ImportError("C++ engine is not built. Run ./build_native.sh first.") from exc

ROOT = Path(__file__).resolve().parent


def package_library(package, version, relative):
    installed = metadata.version(package)
    if installed != version:
        raise RuntimeError(f"Native ABI requires {package}=={version}; found {installed}")
    spec = util.find_spec(package)
    library = Path(spec.origin).parent / relative
    if not library.is_file():
        raise FileNotFoundError(f"Native runtime library not found: {library}")
    return str(library)


class NativeInference:
    def __init__(self, model="mediapipe", face_details=True):
        self._tasks = self._movenet = None
        library = package_library("mediapipe", "1.0.1", "tasks/c/libmediapipe.so")
        assets = ROOT / "pose_probe_assets"
        try:
            if model == "mediapipe" or face_details:
                self._tasks = NativeLandmarkers(
                    library,
                    str(assets / "pose_landmarker_lite.task") if model == "mediapipe" else "",
                    str(assets / "face_landmarker.task") if face_details else "",
                )
            if model == "movenet":
                from _motion_native import NativeMoveNet
                self._movenet = NativeMoveNet(
                    package_library("ai_edge_litert", "2.2.0", "libLiteRt.so"),
                    str(assets / "movenet_lightning_int8_v4.tflite"),
                )
        except BaseException:
            self.close()
            raise

    def infer(self, frame, timestamp):
        points, face = self._tasks.infer(frame, timestamp) if self._tasks is not None else (None, None)
        if self._movenet is not None:
            points = self._movenet.infer(frame, timestamp)
        return points, face

    def close(self):
        for resource in (self._movenet, self._tasks):
            if resource is not None:
                resource.close()
