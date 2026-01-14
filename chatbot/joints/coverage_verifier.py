
from typing import Dict, List
from .base import debug_print

class CoverageVerifierJoint:
    """
    Joint 2.5: Coverage Verification
    
    Checks if the selected articles cover all required entities.
    For comparison queries, ensures each entity has at least one article.
    If gaps are found, suggests targeted search terms.
    """
    
    def __init__(self):
        debug_print("JOINT2.5:INIT", "CoverageVerifier initialized (no LLM required)")
    
    def verify_coverage(
        self, 
        entity_info: Dict, 
        selected_articles: List[Dict]
    ) -> Dict:
        """
        Verify that selected articles cover all necessary entities.
        """
        entities = entity_info.get('entities', [])
        entity_names = [e.get('name', '').strip() for e in entities if e.get('name')]
        
        entity_aliases = {}
        for e in entities:
            name = e.get('name', '').strip()
            if name:
                entity_aliases[name] = [a.strip().lower() for a in e.get('aliases', []) if a]
        
        debug_print("JOINT2.5:VERIFY", f"Checking coverage for {len(entity_names)} entities")
        
        article_titles = []
        for article in selected_articles:
            title = article.get('metadata', {}).get('title', '')
            if title:
                article_titles.append(title.lower().strip())
        
        covered = []
        missing = []
        
        for entity_name in entity_names:
            entity_lower = entity_name.lower()
            aliases = entity_aliases.get(entity_name, [])
            
            found = False
            for title in article_titles:
                if entity_lower in title:
                    found = True
                    break
                for alias in aliases:
                    if alias in title:
                        found = True
                        break
                if found:
                    break
            
            if found:
                covered.append(entity_name)
            else:
                missing.append(entity_name)
        
        suggested_searches = []
        for entity_name in missing:
            suggested_searches.append(entity_name)
            suggested_searches.append(f"{entity_name} (programming language)")
            suggested_searches.append(f"{entity_name} (software)")
            suggested_searches.append(f"{entity_name} (technology)")
            suggested_searches.append(f"{entity_name} (person)")
            
            for e in entities:
                if e.get('name') == entity_name:
                    entity_type = e.get('type', '').lower()
                    if entity_type in ['technology', 'concept']:
                        suggested_searches.append(f"{entity_name} technology")
                    elif entity_type == 'event':
                        suggested_searches.append(f"{entity_name} incident")
                    elif entity_type == 'person':
                        suggested_searches.append(f"{entity_name} biography")
                    break
        
        result = {
            'complete': len(missing) == 0,
            'covered': covered,
            'missing': missing,
            'suggested_searches': suggested_searches[:12]
        }
        
        return result
