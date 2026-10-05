// Deterministic C ABI fault injection for ownership/cleanup tests only.
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include <cstdlib>
#include <cstring>

struct MpPoseLandmarkerInternal {};
struct MpFaceLandmarkerInternal {};
struct MpImageInternal {};
namespace {
int tasks = 0, images = 0, errors = 0;
bool fail_face_create = false, fail_face_close = false, fail_pose_infer = false;
MpStatus fail(char** message) {
  *message = ::strdup("injected MediaPipe failure");
  ++errors;
  return kMpInternal;
}
}
extern "C" {
int MotionTestTasks() { return tasks; }
int MotionTestImages() { return images; }
int MotionTestErrors() { return errors; }
void MotionTestFailFaceCreate() { fail_face_create = true; }
void MotionTestFailFaceClose() { fail_face_close = true; }
void MotionTestFailPoseInfer() { fail_pose_infer = true; }
void MpErrorFree(char* error) { if (error) { std::free(error); --errors; } }
MpStatus MpImageCreateFromUint8Data(MpImageFormat, int, int, const uint8_t*, int,
                                   MpImagePtr* image, char**) {
  *image = new MpImageInternal; ++images; return kMpOk;
}
void MpImageFree(MpImagePtr image) { delete image; --images; }
MpStatus MpPoseLandmarkerCreate(MpPoseLandmarkerOptions*, MpPoseLandmarkerPtr* task, char**) {
  *task = new MpPoseLandmarkerInternal; ++tasks; return kMpOk;
}
MpStatus MpFaceLandmarkerCreate(MpFaceLandmarkerOptions*, MpFaceLandmarkerPtr* task, char** error) {
  if (fail_face_create) { fail_face_create = false; return fail(error); }
  *task = new MpFaceLandmarkerInternal; ++tasks; return kMpOk;
}
MpStatus MpPoseLandmarkerDetectForVideo(MpPoseLandmarkerPtr, MpImagePtr,
    const MpImageProcessingOptions*, int64_t, MpPoseLandmarkerResult*, char** error) {
  if (fail_pose_infer) { fail_pose_infer = false; return fail(error); }
  return kMpOk;
}
MpStatus MpFaceLandmarkerDetectForVideo(MpFaceLandmarkerPtr, MpImagePtr,
    const MpImageProcessingOptions*, int64_t, MpFaceLandmarkerResult*, char**) { return kMpOk; }
void MpPoseLandmarkerCloseResult(MpPoseLandmarkerResult*) {}
void MpFaceLandmarkerCloseResult(MpFaceLandmarkerResult*) {}
MpStatus MpPoseLandmarkerClose(MpPoseLandmarkerPtr task, char**) {
  delete task; --tasks; return kMpOk;
}
MpStatus MpFaceLandmarkerClose(MpFaceLandmarkerPtr task, char** error) {
  if (fail_face_close) { fail_face_close = false; return fail(error); }
  delete task; --tasks; return kMpOk;
}
}
