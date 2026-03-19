import asyncio, logging, os, json
from core.llm_clients import cerebras, groq_client, openrouter, get_gemini
from core.llm_router  import ROUTING_TABLE, NO_LLM_AGENTS, Provider, ModelConfig
import core.llm_cache as cache

logger = logging.getLogger(__name__)
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY  = float(os.getenv("RETRY_DELAY_SECONDS", "2"))

# Errors that mean "skip this model entirely, don't retry"
SKIP_ERRORS = ["402", "401", "403", "insufficient credits",
               "api key", "unauthorized", "not found"]

# Errors that mean "wait and retry same model"
RETRY_ERRORS = ["429", "rate limit", "timeout", "503", "502",
                "overloaded", "service unavailable"]


async def call_llm(
    agent:      str,
    prompt:     str,
    system:     str  = None,
    use_cache:  bool = True,
    pdf_bytes:  bytes = None,   # only for carbon_extraction
) -> str:
    """
    Single entry point for all LLM calls.
    
    Args:
        agent:      Agent name — must match a key in ROUTING_TABLE
        prompt:     User prompt
        system:     Optional system prompt
        use_cache:  Use disk cache (default True)
        pdf_bytes:  Raw PDF bytes — only needed for carbon_extraction
    
    Returns:
        str: Model response
    
    Raises:
        ValueError:   If agent is in NO_LLM_AGENTS or not in ROUTING_TABLE
        RuntimeError: If all providers in the chain fail
    """

    # Guard: never call LLM for pipeline/IO agents
    if agent in NO_LLM_AGENTS:
        raise ValueError(
            f"Agent '{agent}' should never call an LLM. "
            f"It is a pure Python/IO agent. Remove the LLM call."
        )

    # Guard: unknown agent
    if agent not in ROUTING_TABLE:
        raise ValueError(
            f"Agent '{agent}' has no routing entry. "
            f"Add it to ROUTING_TABLE in llm_router.py."
        )

    # Cache check
    if use_cache:
        hit = cache.get(agent, prompt)
        if hit:
            logger.debug("Cache hit: agent=%s", agent)
            return hit

    model_chain = ROUTING_TABLE[agent]
    last_error  = None

    for config in model_chain:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info("Trying %s/%s for agent=%s (attempt %d)",
                            config.provider.value, config.model_id,
                            agent, attempt + 1)

                result = await _dispatch(config, prompt, system, pdf_bytes)

                if use_cache:
                    cache.set(agent, prompt, result)

                logger.info("Success: %s/%s agent=%s",
                            config.provider.value, config.model_id, agent)
                return result

            except Exception as e:
                last_error = e
                err = str(e).lower()

                if any(x in err for x in SKIP_ERRORS):
                    logger.warning("Skip error on %s/%s: %s — next model",
                                   config.provider.value, config.model_id, e)
                    break  # don't retry, jump to next model

                if any(x in err for x in RETRY_ERRORS):
                    wait = RETRY_DELAY * (2 ** attempt)  # exponential backoff
                    logger.warning("Rate limit on %s/%s — waiting %.1fs",
                                   config.provider.value, config.model_id, wait)
                    await asyncio.sleep(wait)
                    continue

                logger.error("Unexpected error %s/%s: %s",
                             config.provider.value, config.model_id, e)
                break

    raise RuntimeError(
        f"All providers failed for agent='{agent}'. "
        f"Chain tried: {[c.model_id for c in model_chain]}. "
        f"Last error: {last_error}"
    )


async def _dispatch(config: ModelConfig, prompt: str,
                    system: str, pdf_bytes: bytes) -> str:

    if config.provider == Provider.GEMINI:
        return await _call_gemini(config, prompt, system, pdf_bytes)
    elif config.provider == Provider.GROQ:
        return await _call_groq(config, prompt, system)
    elif config.provider == Provider.CEREBRAS:
        return await _call_cerebras(config, prompt, system)
    elif config.provider == Provider.OPENROUTER:
        return await _call_openrouter(config, prompt, system)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


async def _call_gemini(config, prompt, system, pdf_bytes) -> str:
    model = get_gemini(config.model_id)
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    if config.json_mode:
        full_prompt += "\n\nReturn valid JSON only. No markdown, no explanation."

    content = []
    if pdf_bytes:
        content.append({"mime_type": "application/pdf", "data": pdf_bytes})
    content.append(full_prompt)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: model.generate_content(content)
    )
    text = response.text.strip()
    # Strip markdown fences that models sometimes add despite instructions
    if config.json_mode and text.startswith("```"):
        text = "\n".join(
            l for l in text.split("\n")
            if not l.strip().startswith("```")
        ).strip()
    return text


async def _call_groq(config, prompt, system) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=config.model_id,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await groq_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


async def _call_cerebras(config, prompt, system) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=config.model_id,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await cerebras.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


async def _call_openrouter(config, prompt, system) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=config.model_id,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await openrouter.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()
