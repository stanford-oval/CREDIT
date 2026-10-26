"""OpenAI-compatible chat-completions client with a disk cache, concurrency, and a daily spend cap.

The endpoint and key come from the environment (CREDIT_API_BASE, CREDIT_API_KEY) or from an env file at
$CREDIT_SECRETS (default $CREDIT_ROOT/.secrets/api.env) with the same two keys. Any OpenAI-compatible server works
(e.g., a LiteLLM proxy in front of several providers). The key is never logged or written.
"""
import hashlib
import re
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
_lock = threading.Lock()

# ---- Daily spend cap (set API_DAILY_USD) --------------------------------------------------------------------------
# Cost is ESTIMATED from token usage × conservative list prices ($ per 1M tokens: input, output incl. reasoning).
# Prices are rough list prices and err high on purpose; adjust PRICES for your provider.
DAILY_USD = float(os.environ.get("API_DAILY_USD", "100"))
PRICES = [  # (model-name regex, $/M input, $/M output)
    (r"gemini-.*flash-lite", 0.15, 1.0), (r"gemini-.*flash", 0.5, 4.0), (r"gemini-.*pro", 2.5, 15.0),
    (r"gpt-5.*nano|gpt-6.*nano", 0.1, 0.8), (r"gpt-5.*mini|gpt-6.*mini", 0.4, 3.0), (r"gpt-5|gpt-6", 2.0, 15.0),
    (r"claude-haiku", 1.0, 5.0), (r"claude-opus", 15.0, 75.0), (r"claude", 3.0, 15.0),
]


class BudgetExceeded(RuntimeError):
    pass


def _price(model):
    for rx, pi, po in PRICES:
        if re.search(rx, model):
            return pi, po
    return 5.0, 25.0  # unknown model: assume expensive


def _ledger_path():
    root = os.environ.get("CREDIT_ROOT", os.path.expanduser("~/.cache"))
    d = os.path.join(root, "api_spend")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, time.strftime("%Y-%m-%d") + ".jsonl")


_spent = {"day": None, "usd": 0.0, "read_at": 0.0}


def _reset_time():
    """Optional budget reset: api_spend/reset_<day>.txt holds a unix time; only spend after it counts toward the cap.
    The ledger itself is never modified (total spend stays auditable)."""
    p = os.path.join(os.path.dirname(_ledger_path()), "reset_" + time.strftime("%Y-%m-%d") + ".txt")
    try:
        return float(open(p).read().strip())
    except Exception:
        return 0.0


def spent_today(refresh=False):
    """Estimated $ spent today (since any reset) across all processes (ledger re-read at most every 20 s)."""
    day = time.strftime("%Y-%m-%d")
    if refresh or _spent["day"] != day or time.time() - _spent["read_at"] > 20:
        tot, t0 = 0.0, _reset_time()
        p = _ledger_path()
        if os.path.exists(p):
            for l in open(p):
                try:
                    x = json.loads(l)
                    if x.get("t", 0) >= t0:
                        tot += x["usd"]
                except Exception:
                    pass
        _spent.update(day=day, usd=tot, read_at=time.time())
    return _spent["usd"]


def _record(model, usage):
    pi, po = _price(model)
    usd = (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6
    _spent["usd"] += usd
    try:
        with open(_ledger_path(), "a") as f:
            f.write(json.dumps({"t": time.time(), "model": model, "usd": round(usd, 6),
                                "in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0)}) + "\n")
    except OSError:
        pass
    if _spent["usd"] > 0.8 * DAILY_USD and not _spent.get("warned"):
        _spent["warned"] = True
        print(f"[api] WARNING: estimated spend today ${_spent['usd']:.2f} > 80% of ${DAILY_USD:.0f} cap", flush=True)


def _creds():
    base, key = os.environ.get("CREDIT_API_BASE"), os.environ.get("CREDIT_API_KEY")
    if base and key:
        return base.rstrip("/"), key
    p = os.environ.get("CREDIT_SECRETS") or os.path.join(os.environ.get("CREDIT_ROOT", "."), ".secrets/api.env")
    if os.path.exists(p):
        d = dict(l.strip().split("=", 1) for l in open(p) if "=" in l and not l.lstrip().startswith("#"))
        return d["CREDIT_API_BASE"].rstrip("/"), d["CREDIT_API_KEY"]
    raise RuntimeError("API credentials not found: set CREDIT_API_BASE and CREDIT_API_KEY (see .env.example)")

def _cache_path():
    root = os.environ.get("CREDIT_ROOT", os.path.expanduser("~/.cache"))
    return os.path.join(root, "api_cache.jsonl")


_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        p = _cache_path()
        if os.path.exists(p):
            for l in open(p):
                try:
                    d = json.loads(l)
                    _cache[d["k"]] = d["v"]
                except Exception:
                    pass
    return _cache


def complete(prompt, model=None, max_tokens=4000, temperature=0.0, system=None, retries=6):
    """Single chat completion. Returns text ('' on failure). Cached on (model, system, prompt, max_tokens, temperature)."""
    import requests

    model = model or JUDGE_MODEL
    key = hashlib.sha256(json.dumps([model, system, prompt, max_tokens, temperature]).encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    if spent_today() >= DAILY_USD:
        raise BudgetExceeded(f"API daily cap reached: est. ${spent_today():.2f} >= ${DAILY_USD:.0f}. Ask the user before raising API_DAILY_USD.")
    base, api_key = _creds()
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if not re.match(r"(gpt-5|gpt-6|o\d)", model):  # reasoning OpenAI models only accept the default temperature
        body["temperature"] = temperature
    out = ""
    for a in range(retries):
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=300)
            if r.status_code == 200:
                js = r.json()
                out = js["choices"][0]["message"].get("content") or ""
                with _lock:
                    _record(model, js.get("usage") or {})
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** a * 2))
                continue
            print(f"[api] {model} HTTP {r.status_code}: {r.text[:200]}", flush=True)
            break
        except Exception as e:  # noqa
            time.sleep(min(60, 2 ** a * 2))
    if out:
        with _lock:
            cache[key] = out
            try:  # a full disk must not kill the job; caching is best-effort
                with open(_cache_path(), "a") as f:
                    f.write(json.dumps({"k": key, "v": out}) + "\n")
            except OSError as e:
                print(f"[api] cache write skipped: {e}", flush=True)
    return out


def batch(prompts, model=None, max_tokens=4000, temperature=0.0, system=None, workers=32):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: complete(p, model, max_tokens, temperature, system), prompts))
