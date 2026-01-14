
import time
from typing import Dict, List, Any
from chatbot import config
from .base import debug_print, local_inference, extract_json_from_text

class EntityExtractorJoint:
    """
    Joint 1: Entity Extraction
    
    Extracts the main entity, type, action, and aliases from a user query.
    Uses llama3.2:1b for fast, focused entity recognition.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.ENTITY_JOINT_MODEL
        self.temperature = config.ENTITY_JOINT_TEMP
        debug_print("JOINT1:INIT", f"EntityExtractor initialized with {self.model}")
    
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract ALL entities from query, with comparison detection.
        
        Args:
            query: User query string
            
        Returns:
            Dict with keys: is_comparison, entities (list), action
            Each entity has: name, type, aliases
        """
        debug_print("JOINT1:ENTITY", f"Extracting entities from: '{query}'")
        start_time = time.time()
        
        prompt = f"""You are a precise entity extraction system optimized for Wikipedia article matching.

INSTRUCTIONS:
1. Identify ALL distinct entities (people, places, things, events) in the query.
2. For each entity, provide the NAME as it would appear as a Wikipedia article title.
3. CHECK FOR COMPARISONS: If the user compares items (e.g. "vs", "compare", "difference", "which is"), set "is_comparison": true.
4. EXTRACT ALIASES: Include alternative names AND related Wikipedia article titles the entity might appear under.
5. IDENTIFY ANSWER TYPE: What specific information does the user want? See examples below.
6. FOR COMPARISONS: Extract the comparison_dimension - what aspect are they comparing on?

ANSWER TYPE EXAMPLES (learn the pattern, not just keywords):
- "When was X born?" → birthdate
- "Where was X born?" → birthplace  
- "What school did X attend?" → education
- "Where did X study?" → education
- "What degree does X have?" → education
- "Who invented X?" → inventor
- "Who created X?" → inventor
- "When did X die?" → death_date
- "How did X die?" → death_cause
- "What language did X speak/write?" → language
- "How tall is X?" → measurement
- "How big is X?" → measurement
- "What caused X?" → cause
- "Why did X happen?" → cause
- General questions → general

COMPARISON DIMENSION EXAMPLES:
- "which came first" → creation_date
- "which is older" → age
- "which is larger/bigger" → size
- "which is taller" → height
- "which is faster" → speed
- "who had more X" → quantity
- "which was more successful" → success

EXAMPLES:
Query: "Who created Python?"
Result: {{
  "is_comparison": false,
  "entities": [
    {{"name": "Python (programming language)", "type": "technology", "aliases": ["Python"]}},
    {{"name": "creator of Python", "type": "person", "aliases": []}}
  ],
  "action": "identify the creator"
}}

Query: "Compare Tesla and Edison patents"
Result: {{
  "is_comparison": true,
  "entities": [
    {{"name": "Nikola Tesla", "type": "person", "aliases": ["Tesla"]}},
    {{"name": "Thomas Edison", "type": "person", "aliases": ["Edison"]}}
  ],
  "action": "compare patent counts",
  "comparison_dimension": "quantity"
}}

Query: "What university did the creator of Python attend?"
Result: {{
  "is_comparison": false,
  "entities": [
    {{"name": "Python (programming language)", "type": "technology", "aliases": ["Python"]}},
    {{"name": "creator of Python", "type": "person", "aliases": []}}
  ],
  "action": "identify the university attended"
}}

WIKIPEDIA TITLE CONVENTIONS:
- Full names for people: "Albert Einstein" not "Einstein"
- Specific names for events: "World War II" not "the war"
- Disambiguation when needed: "Java (programming language)" for the language
- For indirect queries (e.g., "who created X"), extract BOTH the person AND the thing created. Do NOT guess the person's name yet.

Query: "{query}"

CRITICAL RULES:
- Return ONLY valid JSON.
- NO Markdown code blocks.
- Entity names should match Wikipedia article titles exactly if mentioned.
- Do NOT try to answer the query. 
- Do NOT resolve entities to specific names (like 'University of Cambridge') if they aren't explicitly in the query. For 'the creator of X', extract 'creator of X' or similar, not your guess of who it is.
- Include short aliases that might also be article titles.

Return this exact JSON structure:
{{
  "is_comparison": false,
  "entities": [
    {{"name": "Exact Wikipedia Article Title", "type": "person|place|event|concept|technology|organization", "aliases": ["Alternative Title", "Short Form"]}}
  ],
  "action": "what the user wants to know",
  "answer_type": "birthdate|birthplace|education|inventor|death_date|death_cause|language|measurement|cause|general",
  "comparison_dimension": "null or: creation_date|age|size|height|speed|quantity|success"
}}
"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT1:ENTITY", f"Raw response: {response[:300]}...")
            
            # Use robust extractor
            result = extract_json_from_text(response)
            
            # Handle if model returns a list wrapper
            if isinstance(result, list):
                if result:
                    result = result[0]
                else:
                    raise ValueError("Received empty list from model")
            
            # Validate new result structure
            if 'entities' not in result:
                # Try to convert old format to new format
                if 'entity' in result:
                    debug_print("JOINT1:ENTITY", "Converting old format to new multi-entity format")
                    result = {
                        'is_comparison': False,
                        'entities': [{
                            'name': result.get('entity', query),
                            'type': result.get('entity_type', 'unknown'),
                            'aliases': result.get('aliases', [])
                        }],
                        'action': result.get('action', 'information')
                    }
                else:
                    raise ValueError(f"Missing 'entities' key. Got: {result.keys()}")
            
            # Ensure entities is a list
            if not isinstance(result.get('entities'), list):
                raise ValueError(f"'entities' must be a list, got {type(result.get('entities'))}")
            
            # Ensure each entity has required keys
            for i, entity in enumerate(result['entities']):
                if 'name' not in entity:
                    raise ValueError(f"Entity {i} missing 'name' key")
                # Set defaults for optional fields
                entity.setdefault('type', 'unknown')
                entity.setdefault('aliases', [])
            
            elapsed = time.time() - start_time
            entity_names = [e['name'] for e in result['entities']]
            debug_print("JOINT1:ENTITY", f"Extracted {len(result['entities'])} entities: {entity_names}")
            debug_print("JOINT1:ENTITY", f"Is comparison: {result.get('is_comparison', False)}")
            debug_print("JOINT1:ENTITY", f"Action: {result.get('action', 'N/A')}")
            debug_print("JOINT1:ENTITY", f"Extraction took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            debug_print("JOINT1:ENTITY", f"Extraction failed: {type(e).__name__}: {e}")
            # Fallback: return query as single entity in new format
            debug_print("JOINT1:ENTITY", "Using fallback: query as single entity")
            return {
                "is_comparison": False,
                "entities": [{
                    "name": query,
                    "type": "unknown",
                    "aliases": []
                }],
                "action": "information"
            }
            
    def suggest_expansion(self, query: str, failed_terms: List[str]) -> List[str]:
        """
        Suggest alternative search terms when initial search fails.
        
        Args:
            query: User's original query
            failed_terms: List of terms that returned no results
            
        Returns:
            List of new search terms (strings)
        """
        debug_print("JOINT1:EXPAND", f"Suggesting expansion for '{query}' (failed: {failed_terms})")
        start_time = time.time()
        
        prompt = f"""The user asked about: "{query}"
        
        We searched for these terms but found NOTHING relevant: {failed_terms}
        
        INSTRUCTIONS:
        1. Suggest 3 alternative search queries.
        2. Focus on broader concepts, related events, or key figures.
        3. If the user used a nickname, try the real name.
        4. If the user asked a specific question, try searching for the general topic.
        
        Return ONLY a JSON list of strings:
        ["Alternative 1", "Alternative 2", "Alternative 3"]
        """
        
        try:
            response = local_inference(self.model, prompt, temperature=0.3, timeout=config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT1:EXPAND", f"Raw response: {response[:200]}...")
            
            suggestions = extract_json_from_text(response)
            
            if isinstance(suggestions, list):
                # Filter out duplicates and empty strings
                filtered = [s for s in suggestions if isinstance(s, str) and s.strip() and s not in failed_terms]
                debug_print("JOINT1:EXPAND", f"Generated {len(filtered)} suggestions: {filtered}")
                return filtered[:3]
                
            return []
            
        except Exception as e:
            debug_print("JOINT1:EXPAND", f"Expansion failed: {e}")
            return []
