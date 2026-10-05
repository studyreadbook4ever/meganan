"""Live C++ engine / real Tk / HTTP / SSE integration check.

Run with the camera and port8765 free, using the same Tcl/Tk environment as
run_meganan.sh. Only aggregate validation results are retained.
"""
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

from motion_api import MotionAPIServer, StateStore
from motion_tracker import MotionWorker, RuntimeSettings
from motion_view import MotionView


def main():
    from motion_privacy import disable_process_dumps
    disable_process_dumps()
    stop = threading.Event()
    settings = RuntimeSettings(6)
    store = StateStore()
    server = MotionAPIServer(store, port=8765).start()
    worker = MotionWorker(SimpleNamespace(engine="cpp", model="mediapipe",
                          face_details=True, device="/dev/video0", fps_limit=6),
                          store, stop, settings)
    view = MotionView(on_close=stop.set, on_fps_limit=settings.set_fps_limit, fps_limit=6)
    samples, failures, streamed = [], [], []
    began = time.perf_counter()
    phase = -1
    last_sequence = -1
    phases = [(0, 6), (7, 12), (14, 1), (17, 6)]
    transitions = []

    def streaming():
        try:
            with urlopen("http://127.0.0.1:8765/api/v1/events", timeout=3) as response:
                for line in response:
                    if line.startswith(b"data:"):
                        state = json.loads(line[5:])
                        streamed.append((state["sequence"], len(state["face_landmarks"])))
                    if stop.is_set():
                        break
        except Exception as exc:
            if not stop.is_set():
                failures.append("SSE: " + str(exc))

    def refresh():
        nonlocal phase, last_sequence
        try:
            elapsed = time.perf_counter() - began
            if stop.is_set() or elapsed >= 24:
                view.close()
                return
            next_phase = max(i for i, (at, limit) in enumerate(phases) if elapsed >= at)
            if next_phase != phase:
                phase = next_phase
                limit = phases[phase][1]
                view._fps_scale.set(limit)
                view.root.tk.call(view._fps_scale.cget("command"), str(limit))
                assert settings.get_fps_limit() == limit
                transitions.append({"elapsed_s": round(elapsed, 3), "limit": limit})
            state = store.snapshot()
            if state["sequence"] != last_sequence:
                last_sequence = state["sequence"]
                assert state["error"] is None, state["error"]
                view.update(state)
                assert "image" not in {view.canvas.type(item) for item in view.canvas.find_all()}
                with urlopen("http://127.0.0.1:8765/api/v1/state", timeout=2) as response:
                    api = json.load(response)
                assert api["sequence"] >= state["sequence"]
                if api["face_tracked"] and api["face_mode"] == "mesh":
                    assert len(api["face_landmarks"]) == 478
                assert not {"rgb", "image", "frame", "pixels"} & api.keys()
                samples.append({**{k: state[k] for k in
                    ("sequence", "fps", "fps_limit", "face_tracked", "body_tracked", "status")},
                    "settled": elapsed - phases[phase][0] > 1.5})
            view.root.after(30, refresh)
        except BaseException as exc:
            failures.append(repr(exc))
            stop.set()
            view.close()

    stream = threading.Thread(target=streaming, daemon=True)
    try:
        worker.start()
        stream.start()
        view.root.after(0, refresh)
        view.root.mainloop()
    finally:
        stop.set()
        worker.join(timeout=6)
        store.close()
        server.close()
        stream.join(timeout=3)
        view.close()
    assert not worker.is_alive(), "Worker did not terminate"
    assert not stream.is_alive(), "SSE client did not terminate"
    assert not failures, failures
    assert len(transitions) == 4, transitions
    assert len(samples) > 50, "No sustained native states"
    assert len(streamed) > 20, "No sustained SSE output"
    assert any(count == 478 for _, count in streamed), "No face detected in the live check"
    assert all(a[0] < b[0] for a, b in zip(streamed, streamed[1:])), "SSE sequence regressed"
    observed = {}
    for limit in (1, 6, 12):
        fps = [s["fps"] for s in samples
               if s["settled"] and s["fps_limit"] == limit and s["fps"] > 0]
        assert fps and max(fps) <= limit * 1.2, (limit, fps)
        observed[str(limit)] = {"min": min(fps), "max": max(fps), "updates": len(fps)}
    # The native capture thread must release the actual device on shutdown.
    from _motion_native import NativeCamera
    camera = NativeCamera().start()
    try:
        assert camera.next(0, timeout=2) is not None
    finally:
        camera.close()
    report = {"engine": "cpp", "gui_updates": len(samples), "sse_events": len(streamed),
              "face_mesh_updates": sum(s["face_tracked"] for s in samples),
              "fps_slider_changes": transitions, "observed_fps": observed,
              "camera_reopened_after_close": True, "image_canvas_items": 0,
              "errors": failures, "run_summary": str(worker.run_dir / "summary.json")}
    path = Path("artifacts/cpp_desktop_verification.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
