import numpy as np
from face_analysis import analyze_face


def compare_faces(before, after):
    a = analyze_face(before)
    b = analyze_face(after)

    if not a["detected"] or not b["detected"]:
        return {
            "status": "Face not detected",
            "landmark_distance": None,
        }

    points_a = np.array([
        [p["x"], p["y"]] for p in a["landmarks"]
    ], dtype=np.float64)

    points_b = np.array([
        [p["x"], p["y"]] for p in b["landmarks"]
    ], dtype=np.float64)

    # Translate and normalize facial geometry by face scale.
    def normalize(points):
        center = points.mean(axis=0)
        centered = points - center
        scale = np.sqrt(
            np.mean(np.sum(centered ** 2, axis=1))
        )
        if scale < 1e-8:
            raise ValueError("Invalid facial geometry")
        return centered / scale

    a_norm = normalize(points_a)
    b_norm = normalize(points_b)

    distance = np.linalg.norm(
        a_norm - b_norm, axis=1
    ).mean()

    return {
        "status": "OK",
        "landmark_distance": round(float(distance), 4),
        "note": (
            "Lower distance indicates more similar "
            "normalized facial geometry. This is not "
            "identity verification."
        ),
    }
