import gradio as gr

from face_comparison import compare_faces
from pose_analysis import compare_pose


def evaluate(before, after):
    if before is None or after is None:
        raise gr.Error("Please upload both images.")

    before = before.convert("RGB")
    after = after.convert("RGB")

    face_result = compare_faces(before, after)
    pose_result = compare_pose(before, after)

    report = {
        "face_analysis": face_result,
        "body_pose_analysis": pose_result,
        "disclaimer": (
            "Exploratory image consistency analysis. "
            "Results do not prove identity preservation "
            "or anatomical correctness."
        ),
    }

    return before, after, report


with gr.Blocks(title="Fashion Identity QA") as demo:
    gr.Markdown(
        "# Fashion Identity QA\n"
        "Facial Geometry & Body Pose Comparison"
    )

    gr.Markdown(
        "Compare facial landmarks and body pose "
        "between an original and a styled image."
    )

    with gr.Row():
        before = gr.Image(
            type="pil",
            label="Original Person",
        )
        after = gr.Image(
            type="pil",
            label="Generated / Styled Person",
        )

    analyze = gr.Button(
        "Analyze Images",
        variant="primary",
    )

    with gr.Row():
        original_output = gr.Image(
            label="Original Image",
        )
        generated_output = gr.Image(
            label="Styled Image",
        )

    report = gr.JSON(label="Analysis Report")

    analyze.click(
        fn=evaluate,
        inputs=[before, after],
        outputs=[
            original_output,
            generated_output,
            report,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
