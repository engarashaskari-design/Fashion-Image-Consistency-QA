
import numpy as np
import mediapipe as mp


# Joint angles used for body-pose comparison.
# Each tuple contains (point A, joint center, point C).
JOINT_ANGLES = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
}


def extract_pose(image):
    """
    Extract MediaPipe body landmarks from an image.
    Returns 33 landmarks with x, y and visibility.
    """
    if image is None:
        return None

    rgb = np.asarray(image.convert("RGB"))
    rgb = np.ascontiguousarray(rgb)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    return np.array(
        [
            [p.x, p.y, p.visibility]
            for p in result.pose_landmarks.landmark
        ],
        dtype=np.float64,
    )


def calculate_angle(points, indices, image_size):
    """
    Calculate a 2D joint angle in degrees.
    Pixel coordinates are used to account for
    image aspect ratio.
    """
    a, b, c = indices
    width, height = image_size

    xy = points[:, :2].copy()
    xy[:, 0] *= width
    xy[:, 1] *= height

    vector_1 = xy[a] - xy[b]
    vector_2 = xy[c] - xy[b]

    denominator = (
        np.linalg.norm(vector_1)
        * np.linalg.norm(vector_2)
    )

    if denominator < 1e-8:
        return None

    cosine = np.clip(
        np.dot(vector_1, vector_2) / denominator,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(np.arccos(cosine))
    )


def compare_pose(before, after):
    """
    Compare joint angles between two images.

    This is an exploratory 2D pose comparison,
    not a measure of anatomical correctness.
    """
    points_before = extract_pose(before)
    points_after = extract_pose(after)

    if points_before is None or points_after is None:
        return {
            "status": "Not assessable",
            "reason": "Body pose not detected in both images.",
        }

    differences = {}

    for name, indices in JOINT_ANGLES.items():

        # All three landmarks must be sufficiently
        # visible in both images.
        sufficiently_visible = all(
            points_before[i, 2] >= 0.6
            and points_after[i, 2] >= 0.6
            for i in indices
        )

        if not sufficiently_visible:
            continue

        angle_before = calculate_angle(
            points_before,
            indices,
            before.size,
        )

        angle_after = calculate_angle(
            points_after,
            indices,
            after.size,
        )

        if angle_before is None or angle_after is None:
            continue

        differences[name] = round(
            abs(angle_before - angle_after),
            2,
        )

    if len(differences) < 3:
        return {
            "status": "Not assessable",
            "reason": "Not enough reliable joint angles.",
            "valid_joint_angles": len(differences),
        }

    mean_difference = float(
        np.mean(list(differences.values()))
    )

    return {
        "status": "OK",
        "mean_angle_difference_deg": round(
            mean_difference, 2
        ),
        "joint_angle_differences_deg": differences,
        "valid_joint_angles": len(differences),
        "note": (
            "Exploratory 2D joint-angle comparison. "
            "Lower values indicate more similar "
            "joint angles. This does not verify "
            "body proportions or anatomy."
        ),
    }
