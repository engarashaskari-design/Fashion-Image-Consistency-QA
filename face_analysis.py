import cv2
import mediapipe as mp
import numpy as np


def analyze_face(image_rgb):
    """Detect facial landmarks with a crop-based fallback."""
    if image_rgb is None:
        return {"detected": False, "landmarks": []}

    image = np.asarray(image_rgb.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image.")

    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    def run_mesh(region, offset_x=0, offset_y=0,
                 original_w=None, original_h=None):
        rh, rw = region.shape[:2]

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
        ) as mesh:
            result = mesh.process(
                np.ascontiguousarray(region)
            )

        if not result.multi_face_landmarks:
            return None

        points = []
        for p in result.multi_face_landmarks[0].landmark:
            points.append({
                "x": round(offset_x + p.x * rw, 2),
                "y": round(offset_y + p.y * rh, 2),
            })

        return {
            "detected": True,
            "landmarks": points,
            "count": len(points),
        }

    # Attempt 1: original image
    result = run_mesh(image)
    if result is not None:
        result["method"] = "original"
        return result

    # Attempt 2: Haar face detection and crop
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        return {
            "detected": False,
            "landmarks": [],
            "reason": "Face detector unavailable",
        }

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(25, 25),
    )

    if len(faces) == 0:
        return {
            "detected": False,
            "landmarks": [],
            "reason": "No face detected",
        }

    x, y, w, h = max(
        faces, key=lambda f: f[2] * f[3]
    )

    pad_x = int(w * 0.30)
    pad_y = int(h * 0.30)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            "detected": False,
            "landmarks": [],
            "reason": "Invalid face crop",
        }

    # Upscale the crop for small faces.
    ch, cw = crop.shape[:2]
    scale = max(1.0, 512 / max(ch, cw))

    crop_large = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    result = run_mesh(crop_large)

    if result is None:
        return {
            "detected": False,
            "landmarks": [],
            "reason": "FaceMesh failed after cropping",
        }

    # Map landmarks back to the original image.
    for p in result["landmarks"]:
        p["x"] = round(x1 + p["x"] / scale, 2)
        p["y"] = round(y1 + p["y"] / scale, 2)

    result["method"] = "face_crop_fallback"
    return result
