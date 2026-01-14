
import time
import re
from typing import Dict, List, Tuple, Optional
from chatbot import config
from .base import debug_print, local_inference, extract_json_from_text

class ArticleScorerJoint:
    """
    Joint 2: Article Scoring
    
    Scores Wikipedia article titles by relevance to the extracted entity.
    Uses qwen2.5:0.5b for fast scoring.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.SCORER_JOINT_MODEL
        self.temperature = config.SCORER_JOINT_TEMP
        debug_print("JOINT2:INIT", f"ArticleScorer initialized with {self.model}")
    
    def score(self, query: str, entity_info: Dict, article_titles: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Score article titles by relevance to entities and the original query.
        """
        if not article_titles:
            debug_print("JOINT2:SCORER", "No articles to score")
            return []
        
        # Extract all entity names and aliases from new format
        all_entity_names = []
        for entity in entity_info.get('entities', []):
            all_entity_names.append(entity.get('name', ''))
            all_entity_names.extend(entity.get('aliases', []))
        
        # Filter out empty strings
        all_entity_names = [n for n in all_entity_names if n]
        
        # Backwards compatibility: support old format
        if not all_entity_names and 'entity' in entity_info:
            all_entity_names = [entity_info['entity']] + entity_info.get('aliases', [])
        
        entities_display = [e.get('name', '') for e in entity_info.get('entities', [])]
        debug_print("JOINT2:SCORER", f"Scoring {len(article_titles)} articles for entities: {entities_display}")
        start_time = time.time()
        
        # === EXACT MATCH OVERRIDE ===
        exact_match_scores = []
        for title in article_titles:
            title_lower = title.lower().strip()
            if "(disambiguation)" in title_lower:
                continue

            for entity_name in all_entity_names:
                if title_lower == entity_name.lower().strip():
                    debug_print("JOINT2:SCORER", f"EXACT MATCH OVERRIDE: '{title}' == entity '{entity_name}' -> score 11.0")
                    exact_match_scores.append((title, 11.0))
                    break
        
        debug_print("JOINT2:SCORER", f"Found {len(exact_match_scores)} exact entity matches")
        
        articles_formatted = "\n".join([f"{i+1}. {title}" for i, title in enumerate(article_titles[:20])])
        entities_str = ", ".join([f"'{e.get('name', '')}'" for e in entity_info.get('entities', [])])
        
        prompt = f"""I will give you a list of Article Titles.
        
USER'S ORIGINAL QUESTION: "{query}"

Select articles relevant to answering this question.
Entities mentioned: {entities_str}
        
RULES:
1. ONLY select titles from the provided INPUT LIST below.
2. DO NOT output example titles.
3. Output valid JSON only.
4. Prioritize articles that directly answer the user's question.
        
INPUT LIST:
{articles_formatted}
        
Rate each article 0-10 where:
- 10 = Directly relevant to answering the user's question
- 7-9 = Highly relevant to the entities or topic
- 1-6 = Partially relevant or containing related background information
- 0 = Not relevant
        
Return ONLY a JSON array:
[
  {{"title": "Actual Title From List", "score": 10}}
]"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT2:SCORER", f"Raw response: {response[:200]}...")
            
            scores = extract_json_from_text(response)
            
            if isinstance(scores, dict):
                for key in ["items", "scores", "results", "articles"]:
                    if key in scores and isinstance(scores[key], list):
                         scores = scores[key]
                         break
            
            if not isinstance(scores, list):
                 raise ValueError("Response is not a JSON array")

            def normalize_title(t: str) -> str:
                return re.sub(r'[,.:;\'\"-]+', '', t.lower()).strip()
            
            def fuzzy_match(llm_title: str, candidates: List[str]) -> Optional[str]:
                norm_llm = normalize_title(llm_title)
                for candidate in candidates:
                    norm_cand = normalize_title(candidate)
                    if norm_llm == norm_cand:
                        return candidate
                    if norm_cand in norm_llm or norm_llm in norm_cand:
                        return candidate
                return None
            
            valid_titles = list(article_titles)
            placeholder_pattern = re.compile(r'article\s+name|title\s+\d+|example\s+article', re.IGNORECASE)
            
            scored_articles = []
            for item in scores:
                llm_title = item.get('title')
                score = float(item.get('score', 0))
                
                if llm_title in valid_titles:
                    matched_title = llm_title
                else:
                    matched_title = fuzzy_match(llm_title, valid_titles)
                    if not matched_title:
                        continue
                
                if placeholder_pattern.search(matched_title):
                    continue
                
                scored_articles.append((matched_title, score))
            
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            exact_titles = {t for t, _ in exact_match_scores}
            scored_articles = [item for item in scored_articles if item[0] not in exact_titles]
            final_results = exact_match_scores + scored_articles
            
            return final_results[:top_k]
            
        except Exception as e:
            debug_print("JOINT2:SCORER", f"Scoring failed: {e}")
            return [(title, 5.0) for title in article_titles[:top_k]]
