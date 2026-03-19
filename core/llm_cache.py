import os
import json
import hashlib
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "llm_responses")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_TTL_BY_AGENT = {
    "credibility_analysis": 30 * 24 * 60 * 60,  # 30 days
    "peer_comparison":      14 * 24 * 60 * 60,  # 14 days
    "temporal_analysis":     7 * 24 * 60 * 60,  # 7 days
    "default":               7 * 24 * 60 * 60,  # 7 days
}

def _get_cache_path(agent: str, prompt: str) -> str:
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    agent_dir = os.path.join(CACHE_DIR, agent)
    os.makedirs(agent_dir, exist_ok=True)
    return os.path.join(agent_dir, f"{h}.json")

def get(agent: str, prompt: str) -> str | None:
    path = _get_cache_path(agent, prompt)
    if not os.path.exists(path):
        return None
        
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
            
    timestamp = data.get("timestamp", 0)
    ttl = CACHE_TTL_BY_AGENT.get(agent, CACHE_TTL_BY_AGENT["default"])
    
    if time.time() - timestamp > ttl:
        return None
        
    return data.get("response")

def set(agent: str, prompt: str, response: str) -> None:
    path = _get_cache_path(agent, prompt)
    data = {
        "timestamp": time.time(),
        "response": response
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
