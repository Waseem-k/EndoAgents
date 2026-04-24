"""
app.py — Gradio UI for testing the EndoAgents Pipeline
"""

import gradio as gr
from PIL import Image
from loguru import logger
from orchestrator import EndoAgentsOrchestrator

# Initialize globally so the 8GB Gemma model only loads once when the server starts
logger.info("Initializing EndoAgents Orchestrator for UI...")
orchestrator = EndoAgentsOrchestrator()

def process_ultrasound(image: Image.Image, pathology_class: str):
    """Wrapper function for the Gradio interface."""
    if image is None:
        return "Error: Please upload an image first.", False, 0
    
    try:
        # Execute the full LangGraph pipeline
        result = orchestrator.run(image=image, pathology_class=pathology_class)
        return result["caption"], result["passed_judge"], result["retries_used"]
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return f"Pipeline Error: {str(e)}", False, 0

# ── Build the Gradio Interface ───────────────────────────────────────────────

with gr.Blocks(title="EndoAgents 🔬", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 EndoAgents: Multi-Agent Pipeline")
    gr.Markdown("Upload a sagittal TVUS image to test the generation, RAG synthesis, and Judge evaluation loop.")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Ultrasound Image (Upload or Paste)")
            input_class = gr.Radio(
                choices=["Adenomyosis", "Fibroid", "Normal"], 
                value="Adenomyosis", 
                label="Target Pathology Class"
            )
            submit_btn = gr.Button("▶ Run Pipeline", variant="primary")
            
        # Right Column: Outputs
        with gr.Column(scale=1):
            output_caption = gr.Textbox(label="Final Synthesised Caption", lines=12, show_copy_button=True)
            
            with gr.Row():
                output_passed = gr.Checkbox(label="✅ Passed Judge")
                output_retries = gr.Number(label="🔄 Retries Triggered", interactive=False)

    # Wire the button to the function
    submit_btn.click(
        fn=process_ultrasound,
        inputs=[input_image, input_class],
        outputs=[output_caption, output_passed, output_retries]
    )

if __name__ == "__main__":
    logger.info("Launching Gradio UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)