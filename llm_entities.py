"""Task-relevant entity extraction for the Section 3.1 scene graph.

The paper derives the task-relevant object list from the natural-language
instruction with a large language model (GPT-5 in the paper). This module
implements that front-end in a provider-agnostic way: it can call the
Anthropic, OpenAI, or OpenRouter REST API, selected by config or environment,
and it caches the result to disk so pipeline runs are deterministic and do not
re-hit the API on every launch. OpenRouter (OpenAI-compatible) reaches both
OpenAI and Anthropic models through one key; set OPENROUTER_API_KEY and use
namespaced model names such as "openai/gpt-4o-mini" or
"anthropic/claude-3.5-sonnet".

The module deliberately depends only on the Python standard library (HTTP via
``urllib``) so it stays importable and unit-testable without the heavy vision
stack or any vendor SDK.

Typical use from a config (see ``configs/pineapple_llm.yaml``):

    task:
      language_instruction: "pick up the pineapple and place it in the bowl"
      entities:
        source: "llm"        # manual | llm
        provider: "auto"     # auto | anthropic | openai | openrouter
        model: null          # null -> {PROVIDER}_MODEL env, else a default
        max_entities: 6
        cache: true
        extra: []            # optional distractors to always include

or from the command line:

    python llm_entities.py --config configs/pineapple_llm.yaml --write
    python llm_entities.py --instruction "put the mug on the shelf" --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o",
    "openrouter": "openai/gpt-4o-mini",
}
DEFAULT_MAX_ENTITIES = 8

PROMPT_TEMPLATE = (
    "You extract the physical objects relevant to a robot manipulation task.\n"
    "Given the instruction, list the distinct real-world objects (noun phrases) "
    "the robot must perceive to perform it: the objects directly acted on and "
    "any referenced containers or target locations. Exclude the robot itself, "
    "hands, grippers, verbs, and abstract words.\n"
    "Return ONLY a compact JSON array of lowercase noun phrases, most important "
    "first, at most {max_entities} items. No prose, no code fences.\n\n"
    'Instruction: "{instruction}"\n'
)


# --------------------------------------------------------------------------- #
# Provider selection and HTTP
# --------------------------------------------------------------------------- #
def _pick_provider(provider: Optional[str]) -> str:
    """Resolve provider name, using env keys when 'auto' or empty."""
    if provider and provider != "auto":
        return provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    raise RuntimeError(
        "No LLM provider configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
        "OPENROUTER_API_KEY, or set task.entities.provider explicitly."
    )


def _http_post(url: str, headers: Dict[str, str], payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # pragma: no cover - network path
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"{url} returned HTTP {error.code}: {detail}") from error


def _call_anthropic(prompt: str, model: str, api_key: str, timeout: int) -> str:  # pragma: no cover - network
    body = _http_post(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    parts = body.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _call_openai(prompt: str, model: str, api_key: str, timeout: int) -> str:  # pragma: no cover - network
    body = _http_post(
        "https://api.openai.com/v1/chat/completions",
        {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return body["choices"][0]["message"]["content"]


def _call_openrouter(prompt: str, model: str, api_key: str, timeout: int) -> str:  # pragma: no cover - network
    # OpenRouter is OpenAI-compatible; only the base URL, key, and (optional)
    # attribution headers differ. Model names are namespaced, e.g.
    # "openai/gpt-4o-mini" or "anthropic/claude-3.5-sonnet".
    body = _http_post(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_REFERER", "https://github.com/jianingq/scene-graph-robot"
            ),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "scene-graph-robot entity extraction"),
        },
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout,
    )
    return body["choices"][0]["message"]["content"]


def _call_llm(prompt: str, provider: str, model: str, api_key: str, timeout: int) -> str:
    if provider == "anthropic":
        return _call_anthropic(prompt, model, api_key, timeout)
    if provider == "openai":
        return _call_openai(prompt, model, api_key, timeout)
    if provider == "openrouter":
        return _call_openrouter(prompt, model, api_key, timeout)
    raise ValueError(f"Unknown provider: {provider!r}")


# --------------------------------------------------------------------------- #
# Parsing and formatting
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    # Lowercase, collapse whitespace, and strip surrounding punctuation and
    # list-bullet markers ("- ", "* ", "1.") that models sometimes emit.
    cleaned = " ".join(str(name).lower().split()).strip(" .\"'`-*•\t")
    return re.sub(r"^\d+[.)]\s*", "", cleaned).strip()


def _parse_entities(text: str, max_entities: int) -> List[str]:
    """Parse an LLM response into a clean, de-duplicated entity list."""
    match = re.search(r"\[.*\]", text, re.S)
    items: List = []
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = []
    if not items:
        # Fallback: treat the response as a delimited list.
        items = re.split(r"[\n,]", text)

    entities: List[str] = []
    seen = set()
    for item in items:
        name = _normalize(item)
        if name and name not in seen:
            seen.add(name)
            entities.append(name)
    return entities[:max_entities]


def to_grounding_classes(entities: List[str]) -> List[str]:
    """Convert entity phrases to the GroundingDINO caption tokens the pipeline
    expects, e.g. "yellow pineapple" -> "yellow.pineapple." and "bowl" ->
    "bowl." (dots for spaces, trailing dot per class)."""
    return [".".join(_normalize(e).split()) + "." for e in entities]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_entities(
    instruction: str,
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    extra: Optional[List[str]] = None,
    timeout: int = 60,
) -> Tuple[List[str], str, str]:
    """Extract task-relevant noun entities from an instruction via an LLM.

    Returns (entities, provider, model).
    """
    provider = _pick_provider(provider)
    model = model or os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key for provider {provider!r}.")

    prompt = PROMPT_TEMPLATE.format(instruction=instruction, max_entities=max_entities)
    text = _call_llm(prompt, provider, model, api_key, timeout)
    entities = _parse_entities(text, max_entities)

    for item in extra or []:
        name = _normalize(item)
        if name and name not in entities:
            entities.append(name)

    if not entities:
        raise RuntimeError(
            f"LLM returned no usable entities for instruction {instruction!r} "
            f"(raw response: {text!r})."
        )
    return entities, provider, model


def resolve_entities(
    entities_cfg: Optional[dict],
    instruction: str,
    config_dir: str,
    force: bool = False,
) -> Dict:
    """Resolve the entity/class list for a task, using an on-disk cache when
    possible so repeated pipeline runs are deterministic and offline.

    ``entities_cfg`` is the plain dict from ``task.entities``. Returns a dict
    with keys: instruction, provider, model, entities, classes.
    """
    cfg = dict(entities_cfg or {})
    max_entities = int(cfg.get("max_entities", DEFAULT_MAX_ENTITIES))
    extra = list(cfg.get("extra", []) or [])
    use_cache = bool(cfg.get("cache", True))

    cache_name = cfg.get("cache_file") or "entities.json"
    cache_path = (
        cache_name if os.path.isabs(cache_name) else os.path.join(config_dir, cache_name)
    )

    if use_cache and not force and os.path.exists(cache_path):
        with open(cache_path, "r") as handle:
            cached = json.load(handle)
        if cached.get("instruction") == instruction and cached.get("classes"):
            return cached

    entities, provider, model = extract_entities(
        instruction,
        provider=cfg.get("provider", "auto"),
        model=cfg.get("model"),
        max_entities=max_entities,
        extra=extra,
        timeout=int(cfg.get("timeout", 60)),
    )
    result = {
        "instruction": instruction,
        "provider": provider,
        "model": model,
        "entities": entities,
        "classes": to_grounding_classes(entities),
    }
    if use_cache:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


def build_slot_by_class(num_classes: int) -> Dict[int, int]:
    """Assign each class index i to feature slot i+1 (slot 0 is the global scene
    node), matching the compact instance ordering used by the processor."""
    return {i: i + 1 for i in range(num_classes)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_task_fields(config_path: str) -> Tuple[str, dict, str, str]:
    """Minimal defaults+task merge to read instruction and entities config,
    without importing the heavy pipeline module."""
    from omegaconf import OmegaConf  # local import: only needed for --config

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    stem = os.path.splitext(os.path.basename(config_path))[0]
    task_config = OmegaConf.load(config_path)
    merged = OmegaConf.create({})
    for default in task_config.get("defaults", []):
        default_path = os.path.join(config_dir, f"{default}.yaml")
        merged = OmegaConf.merge(merged, OmegaConf.load(default_path))
    merged = OmegaConf.merge(merged, task_config)
    OmegaConf.resolve(merged)
    instruction = merged.task.get("language_instruction")
    entities_cfg = OmegaConf.to_container(merged.task.get("entities") or {}, resolve=True)
    return instruction, entities_cfg, config_dir, stem


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract task-relevant entities (Section 3.1) via an LLM."
    )
    parser.add_argument("--config", help="Task YAML to read the instruction from.")
    parser.add_argument("--instruction", help="Instruction text (overrides --config).")
    parser.add_argument("--provider", default="auto", help="auto | anthropic | openai | openrouter")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-entities", type=int, default=DEFAULT_MAX_ENTITIES)
    parser.add_argument("--force", action="store_true", help="Ignore any cache.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="With --config, refresh and write the entities cache file.",
    )
    args = parser.parse_args(argv)

    if args.instruction:
        entities, provider, model = extract_entities(
            args.instruction,
            provider=args.provider,
            model=args.model,
            max_entities=args.max_entities,
        )
        print(json.dumps({
            "instruction": args.instruction,
            "provider": provider,
            "model": model,
            "entities": entities,
            "classes": to_grounding_classes(entities),
        }, indent=2))
        return 0

    if args.config:
        instruction, entities_cfg, config_dir, stem = _load_task_fields(args.config)
        if not instruction:
            parser.error("task.language_instruction is missing in the config.")
        if args.provider != "auto":
            entities_cfg["provider"] = args.provider
        if args.model:
            entities_cfg["model"] = args.model
        entities_cfg["max_entities"] = args.max_entities
        if args.write and not entities_cfg.get("cache_file"):
            entities_cfg["cache_file"] = f"{stem}.entities.json"
        result = resolve_entities(
            entities_cfg, instruction, config_dir, force=args.force or args.write
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error("Provide --instruction or --config.")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
