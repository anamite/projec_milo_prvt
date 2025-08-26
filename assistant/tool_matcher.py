import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTECE_MODEL_AVAILABLE = True
except Exception:
    SENTECE_MODEL_AVAILABLE = False


class ToolMatcher:
    """Match user input to available tools using embeddings when possible,
    otherwise fallback to simple keyword overlap.

    Logs matching details and errors to aid debugging when matches are
    surprising or absent.
    """

    def __init__(self, tools: List[Dict]):
        self.tools = tools
        self.tool_descriptions = [tool.get('tool_description', '') for tool in tools]
        self.model = None
        self.tool_embeddings = None

        logger.debug(f"Initializing ToolMatcher with {len(tools)} tools; sentence_model_available={SENTECE_MODEL_AVAILABLE}")

        if SENTECE_MODEL_AVAILABLE and len(self.tool_descriptions) > 0:
            try:
                self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                self.tool_embeddings = self.model.encode(self.tool_descriptions)
                logger.info(f"Tool matcher initialized with {len(tools)} tools (embeddings)")
            except Exception:
                logger.exception("Failed to load embedding model for ToolMatcher")
                self.model = None
                self.tool_embeddings = None
        else:
            logger.info("Sentence embedding model unavailable; using keyword matcher")

    def find_best_match(self, user_input: str, threshold: float = 0.3) -> Optional[Dict]:
        logger.debug("find_best_match called", extra={"user_input_preview": (user_input or '')[:160], "threshold": threshold})
        if not user_input:
            logger.debug("Empty user_input passed to find_best_match")
            return None

        # Embedding-based matching when available
        if self.model and self.tool_embeddings is not None:
            try:
                user_embedding = self.model.encode([user_input])
                similarities = cosine_similarity(user_embedding, self.tool_embeddings)[0]
                best_idx = int(similarities.argmax())
                score = float(similarities[best_idx])
                logger.debug(f"Embedding similarities: best_idx={best_idx}, score={score}")
                if score >= threshold:
                    matched = {'tool': self.tools[best_idx], 'confidence': score, 'user_input': user_input}
                    logger.info(f"Embedding match success: {matched.get('tool', {}).get('tool_name')} (score={score})")
                    return matched
                logger.debug("No embedding match above threshold")
                return None
            except Exception:
                logger.exception("Embedding match error")
                return None

        # Fallback: simple keyword overlap between user input and tool description/name
        ui = user_input.lower()
        best = None
        best_score = 0
        for tool in self.tools:
            score = 0
            name = tool.get('tool_name', '').lower()
            desc = tool.get('tool_description', '').lower()
            for token in (name + ' ' + desc).split():
                if token and token in ui:
                    score += 1
            if score > best_score:
                best_score = score
                best = tool

        logger.debug(f"Keyword matching best_score={best_score}")
        if best and best_score > 0:
            matched = {'tool': best, 'confidence': float(best_score), 'user_input': user_input}
            logger.info(f"Keyword match success: {matched.get('tool', {}).get('tool_name')} (score={best_score})")
            return matched
        logger.debug("No keyword match found")
        return None
