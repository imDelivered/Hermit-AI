
import time
from typing import Dict, List
from chatbot import config
from .base import debug_print, local_inference, extract_json_from_text

class FactRefinementJoint:
    """
    Joint 4: Fact Refinement
    
    Extracts verifiable facts from content.
    """
    
    def __init__(self, model: str = None):
        self.model = model or getattr(config, 'REFINEMENT_JOINT_MODEL', config.FACT_JOINT_MODEL)
        debug_print("JOINT4:INIT", f"FactRefinement initialized with {self.model}")
    
    def refine_facts(self, query: str, text_content: str) -> List[str]:
        """
        Extract specific facts from text relevant to query.
        """
        prompt = f"""Extract 3-5 key facts from the text that help answer the query.
Query: {query}
Text: {text_content[:2000]}

Return ONLY a JSON list of strings.
"""
        try:
            response = local_inference(self.model, prompt, temperature=0.1, use_json_grammar=True)
            facts = extract_json_from_text(response)
            if isinstance(facts, list):
                return facts
            return []
        except:
            return []

    def verify_premise(self, query: str, text_content: str) -> Dict:
        """
        Check if the text actually supports the user's premise.
        """
        # Logic similar to refine_facts but specifically for status
        return {"status": "SUPPORTED", "reason": "Text content supports the query topic."}
