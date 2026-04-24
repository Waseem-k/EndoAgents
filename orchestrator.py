"""
orchestrator.py — EndoAgents Pipeline Orchestrator (Day 8-9)
─────────────────────────────────────────────────────────────────────────────
Coordinates the full agent pipeline:
  GemmaVision → NarratorAgent → RAGAgent → CaptionSynthesiser → JudgeAgent

Stub for now. Full implementation in Day 8-9 once all agents are built.

Architecture:
    Image
      │
      ▼
  [GemmaVision]          ← Pre-trained / fine-tuned Gemma 4
      │  raw caption
      ▼
  [NarratorAgent]        ← Enforces 7-section format + self-reflects
      │  structured sections + confidence scores
      ├──────────────────────────────────────────┐
      ▼                                          ▼
  [RAGAgent]                              (parallel)
  Retrieves ISUOG/MUSA passages
      │  relevant clinical context
      └──────────────────────────────────────────┘
                          │
                          ▼
              [CaptionSynthesiser]
              Confidence-weighted merge
                          │ draft caption
                          ▼
                    [JudgeAgent]
              Groundedness / Completeness /
              Clinical Consistency scoring
                          │
              ┌───────────┴────────────┐
              │ pass                   │ fail (≤2 retries)
              ▼                        ▼
        Final Caption          Structured feedback
                                → back to Synthesiser
"""

"""
orchestrator.py — EndoAgents Pipeline Orchestrator (Day 8-9)
─────────────────────────────────────────────────────────────────────────────
Coordinates the full agent pipeline using LangGraph:
  GemmaVision → NarratorAgent → RAGAgent → CaptionSynthesiser ⟷ JudgeAgent
"""

from typing import TypedDict, Optional, List, Any
from PIL import Image
from loguru import logger
from langgraph.graph import StateGraph, END

from config.settings import settings
from models.vision import GemmaVision
from agents.narrator import NarratorAgent, NarratorOutput
from agents.rag_agent import RAGAgent
from agents.synthesiser import CaptionSynthesiser
from agents.judge import JudgeAgent


class EndoAgentsState(TypedDict):
    """LangGraph State dictionary to track the pipeline execution."""
    image: Image.Image
    pathology_class: str
    narrator_output: Optional[NarratorOutput]
    rag_context: List[str]
    draft_caption: str
    judge_feedback: Optional[str]
    retries: int
    final_caption: Optional[str]
    passed: bool


class EndoAgentsOrchestrator:
    """Full LangGraph-powered pipeline orchestrator."""

    def __init__(
        self, 
        linkage: Optional[Any] = None, 
        threshold: Optional[float] = None, 
        max_retries: int = 2
    ) -> None:
        logger.info("EndoAgentsOrchestrator initialised.")
        
        self.linkage = linkage
        self.threshold = threshold if threshold is not None else settings.judge_threshold
        self.max_retries = max_retries
        
        # 1. Load the Vision Model ONCE to share across agents
        logger.info("Loading shared GemmaVision model...")
        self.vision_model = GemmaVision().load()
        
        # 2. Initialize Agents
        self.narrator = NarratorAgent(vision_model=self.vision_model)
        self.rag = RAGAgent()
        self.synthesiser = CaptionSynthesiser(vision_model=self.vision_model)
        self.judge = JudgeAgent(threshold=self.threshold)
        
        # 3. Build and compile the LangGraph workflow
        self.app = self._build_graph()

    def _build_graph(self):
        """Constructs the node and edge topology for the multi-agent system."""
        workflow = StateGraph(EndoAgentsState)
        
        # Add core nodes
        workflow.add_node("narrator", self._narrator_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("synthesiser", self._synthesiser_node)
        workflow.add_node("judge", self._judge_node)
        
        # Define standard linear flow
        workflow.set_entry_point("narrator")
        workflow.add_edge("narrator", "rag")
        workflow.add_edge("rag", "synthesiser")
        workflow.add_edge("synthesiser", "judge")
        
        # Define conditional routing from the judge
        workflow.add_conditional_edges(
            "judge",
            self._judge_router,
            {
                "pass": END,
                "retry": "synthesiser",
                "fail": END
            }
        )
        
        return workflow.compile()

    # ── Node Wrapper Functions ───────────────────────────────────────────────

    def _narrator_node(self, state: EndoAgentsState) -> dict:
        logger.info("-> Node: Narrator")
        output = self.narrator.run(state["image"], state["pathology_class"])
        return {"narrator_output": output}

    def _rag_node(self, state: EndoAgentsState) -> dict:
        logger.info("-> Node: RAG")
        # Ensure compatibility with teammate's upcoming RAGAgent implementation
        # Assuming RAGAgent returns a dict with a "context" list
        rag_result = self.rag.run(state["narrator_output"])
        context = rag_result.get("context", []) if isinstance(rag_result, dict) else []
        return {"rag_context": context}

    def _synthesiser_node(self, state: EndoAgentsState) -> dict:
        logger.info(f"-> Node: Synthesiser (Retry Loop: {state['retries']})")
        output = self.synthesiser.run(
            image=state["image"],
            narrator_output=state["narrator_output"],
            rag_context=state["rag_context"],
            judge_feedback=state["judge_feedback"]
        )
        return {"draft_caption": output.draft_caption}

    def _judge_node(self, state: EndoAgentsState) -> dict:
        logger.info("-> Node: Judge")
        output = self.judge.run(
            draft_caption=state["draft_caption"],
            pathology_class=state["pathology_class"],
            narrator_output=state["narrator_output"],
            rag_context=state["rag_context"]
        )
        
        # Increment retries only if the Judge fails the caption
        new_retries = state["retries"] + 1 if not output.passed else state["retries"]
        
        return {
            "passed": output.passed,
            "judge_feedback": output.feedback if not output.passed else None,
            "retries": new_retries,
            "final_caption": state["draft_caption"] 
        }

    def _judge_router(self, state: EndoAgentsState) -> str:
        """Determines the next step based on Judge output and retry limits."""
        if state["passed"]:
            logger.success("Judge approved the caption. Exiting pipeline.")
            return "pass"
        elif state["retries"] < self.max_retries:
            logger.warning(f"Judge failed. Initiating retry {state['retries']}/{self.max_retries}")
            return "retry"
        else:
            logger.error("Max retries reached. Exiting pipeline with current draft.")
            return "fail"

    # ── Main Execution ───────────────────────────────────────────────────────

    def run(self, image: Image.Image, pathology_class: str = "Adenomyosis") -> dict:
        """
        Entry point to run the orchestrated pipeline for a single image.
        """
        initial_state = {
            "image": image,
            "pathology_class": pathology_class,
            "narrator_output": None,
            "rag_context": [],
            "draft_caption": "",
            "judge_feedback": None,
            "retries": 0,
            "final_caption": None,
            "passed": False
        }
        
        logger.info(f"Starting EndoAgents LangGraph pipeline for class: {pathology_class}")
        final_state = self.app.invoke(initial_state)
        
        # Format the final dictionary output
        narrator_data = final_state.get("narrator_output")
        return {
            "caption": final_state.get("final_caption"),
            "passed_judge": final_state.get("passed"),
            "retries_used": final_state.get("retries"),
            "narrator_confidence": narrator_data.confidence if narrator_data else {}
        }

if __name__ == "__main__":
    # Smoke test stub for the orchestrator
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to test ultrasound image")
    args = parser.parse_args()
    
    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        logger.info("No image provided, creating a dummy gray image...")
        img = Image.new("RGB", (896, 896), color=(127, 127, 127))
        
    orchestrator = EndoAgentsOrchestrator()
    result = orchestrator.run(image=img, pathology_class="Adenomyosis")
    
    print("\n=== PIPELINE RESULTS ===")
    print(f"Passed Judge : {result['passed_judge']}")
    print(f"Retries Used : {result['retries_used']}")
    print("\n=== FINAL CAPTION ===")
    print(result['caption'])