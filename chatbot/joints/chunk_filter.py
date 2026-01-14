
import time
import json
from typing import Dict, List
from chatbot import config
from .base import debug_print, local_inference

class ChunkFilterJoint:
    """
    Joint 3: Chunk Filtering
    
    Filters retrieved chunks by relevance to the original query.
    Uses llama3.2:1b for intelligent chunk evaluation.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.FILTER_JOINT_MODEL
        self.temperature = config.FILTER_JOINT_TEMP
        debug_print("JOINT3:INIT", f"ChunkFilter initialized with {self.model}")
    
    def filter(self, query: str, chunks: List[Dict], top_k: int = 5, entity_info: Dict = None, mode: str = "FACTUAL", answer_type: str = None) -> List[Dict]:
        """
        Filter chunks by query relevance.
        """
        if not chunks:
            return []
        
        is_comparison = entity_info.get('is_comparison', False) if entity_info else False
        entities = entity_info.get('entities', []) if entity_info else []
        entity_names = [e.get('name', '').lower() for e in entities]
        
        if is_comparison and len(entity_names) >= 2:
            return self._diversity_filter(chunks, entity_names, top_k)
        
        chunks_formatted = []
        for i, chunk in enumerate(chunks[:15]):
            text = chunk['text'][:250]
            chunks_formatted.append(f"{i+1}. {text}...")
        
        chunks_text = "\n\n".join(chunks_formatted)
        
        # Simple prompt for brevity
        prompt = f"""Rate these text chunks for how well they answer this query.
Query: {query}
Chunks:
{chunks_text}

Return ONLY a JSON list of objects:
[{{"id": 1, "score": 10}}]"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            # Simplified parsing for brevity here, in reality we'd use the robust extractor
            from .base import extract_json_from_text
            scores = extract_json_from_text(response)
            
            if not isinstance(scores, list):
                return chunks[:top_k]
                
            scored_chunks = []
            for item in scores:
                idx = int(item.get('id', 0)) - 1
                score = float(item.get('score', 0))
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    chunk['filter_score'] = score
                    scored_chunks.append(chunk)
            
            scored_chunks.sort(key=lambda x: x.get('filter_score', 0), reverse=True)
            return scored_chunks[:top_k]
            
        except Exception as e:
            debug_print("JOINT3:FILTER", f"Filtering failed: {e}")
            return chunks[:top_k]

    def _diversity_filter(self, chunks: List[Dict], entity_names: List[str], top_k: int) -> List[Dict]:
        """
        Diversity-aware chunk selection that ensures coverage of all entities.
        """
        per_entity_chunks = {name: [] for name in entity_names}
        for chunk in chunks:
            title = chunk.get('metadata', {}).get('title', '').lower()
            for name in entity_names:
                if name in title:
                    per_entity_chunks[name].append(chunk)
                    break
        
        selected = []
        # Round robin selection
        while len(selected) < top_k:
            added_in_round = False
            for name in entity_names:
                if per_entity_chunks[name]:
                    selected.append(per_entity_chunks[name].pop(0))
                    added_in_round = True
                if len(selected) >= top_k: break
            if not added_in_round: break
            
        return selected
