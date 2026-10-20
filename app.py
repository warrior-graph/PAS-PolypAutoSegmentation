"""Gradio demo.

    pip install gradio
    python app.py --checkpoint runs/emcad_b0/best.pth [--share]
"""

import argparse

import numpy as np

from pas.inference import Predictor, overlay


def main():
    import gradio as gr

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    predictor = Predictor(args.checkpoint, args.device)

    def segment(image, threshold, tta):
        if image is None:
            return None, None, ""
        predictor.tta = tta
        prob = predictor(image)
        mask = ((prob >= threshold) * 255).astype(np.uint8)
        area = 100 * (prob >= threshold).mean()
        return overlay(image, prob, threshold), mask, f"Polyp area: {area:.1f}% of the frame"

    demo = gr.Interface(
        fn=segment,
        inputs=[
            gr.Image(type="numpy", label="Colonoscopy frame"),
            gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Threshold"),
            gr.Checkbox(label="Flip TTA"),
        ],
        outputs=[gr.Image(label="Overlay"), gr.Image(label="Mask"), gr.Textbox(label="Info")],
        title="PAS - Polyp Auto Segmentation",
        description="Research prototype. Not a medical device.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
