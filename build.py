#!/usr/bin/env python3
"""
Autonomous Builder
──────────────────
Accepts any project brief (from a file, or entered interactively).

Phase 0  — Feasibility Check: determines whether the stated goals are
            achievable on the described hardware before a line of code is written.
Phase 1  — Architect: reasons through a full system design, selects models, writes sys_config.json.
Phase 2  — Coder: generates each specified file.
Phase 3  — Verifier: syntax → import → pytest, with auto-fix on failure.

Usage:
    pip install anthropic rich
    export ANTHROPIC_API_KEY=sk-...
    python build.py                          # interactive brief entry
    python build.py --brief my_brief.txt     # load brief from file
    python build.py --brief my_brief.txt --out ./my_project
    python build.py --brief my_brief.txt --skip-feasibility
    python build.py --brief my_brief.txt --force   # proceed even if infeasible
"""

import anthropic
import argparse
import ast
import asyncio
import atexit
import json
import logging
import os
import re
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import box
from rich.padding import Padding
from rich.columns import Columns
from rich.markdown import Markdown

console = Console()

# ── Persistent build logger (Improvement 6) ───────────────────────────────────
# A module-level logger writes detailed state transitions and full LLM prompts /
# responses to build.log.  The rich console UI remains unchanged — logging runs
# in parallel via a FileHandler so it never clutters the terminal.
#
# Log levels used throughout the codebase:
#   DEBUG   — full LLM prompt/response bodies (can be large)
#   INFO    — state transitions: phase starts, file generated, checkpoint saved
#   WARNING — recoverable issues: port conflicts, empty-code recovery, pip errors
#   ERROR   — hard failures that cause a file or build to abort

_build_logger = logging.getLogger("autonomous_builder")
_build_logger.setLevel(logging.DEBUG)
# Prevent propagation to the root logger (avoids duplicate lines in some envs)
_build_logger.propagate = False


def setup_file_logging(out_dir: Path) -> Path:
    """
    Attach a FileHandler to _build_logger that writes to out_dir/build.log.
    Safe to call multiple times — idempotent if the same path is already attached.
    Returns the log file path.
    """
    log_path = out_dir / "build.log"
    # Remove any existing handlers pointing to a different file (e.g. from a
    # previous --out directory in the same process), but keep ones already
    # writing to this exact path.
    for handler in list(_build_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_path.resolve():
                return log_path   # already attached
            _build_logger.removeHandler(handler)
            handler.close()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    _build_logger.addHandler(fh)
    return log_path


def _log_llm_call(label: str, model: str, system: str, user_content: str,
                  response_text: str) -> None:
    """
    Write a full LLM round-trip to the debug log.
    Truncates very long bodies to keep the file manageable (first 4 000 chars).
    """
    MAX_BODY = 4_000
    _build_logger.debug(
        "LLM CALL  label=%s  model=%s\n"
        "  [SYSTEM] %s\n"
        "  [USER]   %s\n"
        "  [REPLY]  %s",
        label, model,
        system[:MAX_BODY],
        user_content[:MAX_BODY],
        response_text[:MAX_BODY],
    )

ARCHITECT_MODEL = "claude-sonnet-4-20250514"
CODER_MODEL     = "claude-sonnet-4-20250514"
TESTER_MODEL    = "claude-sonnet-4-20250514"
MAX_TOKENS      = 8000
MAX_ARCH_ITERS  = 12
MAX_RETRIES     = 3       # attempts per file before giving up and continuing

# ── Approved model registry ─────────────────────────────────────────────────
# Only models that are openly downloadable without HuggingFace gating approval.
# The Architect MUST select from this list. Each entry:
#   model_id  — HuggingFace repo ID or Ollama tag (exact match key)
#   roles     — which roles this model is suitable for
#   runtime   — expected runtime (vllm-rocm, vllm-cuda, ollama, etc.)
#   vram_gb   — typical VRAM at default quantisation (0 = CPU-only)
#   notes     — any extra context for the Architect
#
# To approve a gated model you already have access to, add it here.

APPROVED_MODELS = [
    # ── Vision / VLM (GPU) ──────────────────────────────────────────────────
   
    {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "roles": ["vision"],
        "runtime": "vllm-rocm",
        "vram_gb": 16,
        "notes": "Excellent vision-language, open weights.",
    },
  
    # ── Embedding ───────────────────────────────────────────────────────────
    {
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "roles": ["embedding"],
        "runtime": "vllm-rocm",
        "vram_gb": 4,
        "notes": "Vision+text CLIP embedding, open weights, good for identity matching.",
    },
    
    # ── Coding (CPU / Ollama) ───────────────────────────────────────────────
    {
        "model_id": "qwen2.5-coder:7b",
        "roles": ["coding"],
        "runtime": "ollama",
        "vram_gb": 0,
        "notes": "CPU-based via Ollama. Good code generation, no GPU needed.",
    },
    {
        "model_id": "qwen2.5-coder:3b",
        "roles": ["coding"],
        "runtime": "ollama",
        "vram_gb": 0,
        "notes": "Smaller/faster variant, CPU-only.",
    },
    {
        "model_id": "deepseek-coder-v2:lite",
        "roles": ["coding"],
        "runtime": "ollama",
        "vram_gb": 0,
        "notes": "CPU-based via Ollama. Lightweight code model.",
    },
    {
        "model_id": "codellama:7b",
        "roles": ["coding"],
        "runtime": "ollama",
        "vram_gb": 0,
        "notes": "Meta CodeLlama 7B, open weights, CPU via Ollama.",
    },
    # ── Detection (optional role) ───────────────────────────────────────────
    {
        "model_id": "ultralytics/yolov8",
        "roles": ["detection"],
        "runtime": "direct",
        "vram_gb": 1,
        "notes": "YOLOv8 object detection, pip installable, minimal VRAM.",
    },
]

# Build a fast lookup set for validation
_APPROVED_MODEL_IDS = frozenset(m["model_id"] for m in APPROVED_MODELS)


def _format_approved_models_for_prompt() -> str:
    """Format the approved model registry as a readable table for the system prompt."""
    lines = ["| Model ID | Roles | Runtime | VRAM (GB) | Notes |",
             "|----------|-------|---------|-----------|-------|"]
    for m in APPROVED_MODELS:
        roles = ", ".join(m["roles"])
        vram = str(m["vram_gb"]) if m["vram_gb"] else "CPU"
        lines.append(f"| {m['model_id']} | {roles} | {m['runtime']} | {vram} | {m['notes']} |")
    return "\n".join(lines)


def _validate_model_stack(model_stack: dict) -> list[str]:
    """
    Check every model_id in the Architect's model_stack against APPROVED_MODELS.
    Returns a list of error strings (empty = all approved).
    """
    errors = []
    for role, info in model_stack.items():
        mid = info.get("model_id", "") if isinstance(info, dict) else str(info)
        if mid not in _APPROVED_MODEL_IDS:
            errors.append(
                f"  • {role}: '{mid}' is NOT in the approved model list. "
                f"It may require HuggingFace gating approval or is unknown. "
                f"Pick from: {[m['model_id'] for m in APPROVED_MODELS if role in m['roles']]}"
            )
    return errors


# ── Token budget guard ───────────────────────────────────────────────────────
# Improvement 4 — Precise tokenisation:
#   Use tiktoken (cl100k_base, which closely matches Claude's tokeniser) when
#   the library is installed.  If not installed, fall back to the original
#   3.5 chars/token heuristic.  This makes the budget guard far more accurate
#   for multilingual text, code with many symbols, and dense JSON contracts.

try:
    import tiktoken as _tiktoken
    _TK_ENC = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        """Return token count using tiktoken cl100k_base encoding."""
        return len(_TK_ENC.encode(text))

except ImportError:
    _TK_ENC = None

    def _count_tokens(text: str) -> int:           # type: ignore[misc]
        """Fall back to 3.5 chars/token estimate when tiktoken is not installed."""
        return len(text) // 3


# Fraction of MAX_TOKENS that the prompt (spec + contracts) should not exceed,
# leaving the remainder as headroom for the generated code response.
_PROMPT_BUDGET_FRACTION = 0.65


def _warn_if_spec_too_large(filename: str, spec: str, prior_specs: list[dict]) -> None:
    """
    Improvement 4 — Code Generation token budget guard.

    Estimate whether the full coder prompt (spec + prior contracts) approaches
    the MAX_TOKENS response budget and warn the user if so.  Uses tiktoken for
    precise counting when available, the original char-ratio heuristic otherwise.

    The threshold is MAX_TOKENS * _PROMPT_BUDGET_FRACTION, which represents the
    maximum share of the token budget that should be consumed by the prompt
    (leaving ~35 % for the model's generated code response).
    """
    total_text = spec + "".join(p.get("api_contract", "") for p in prior_specs)
    est_tokens = _count_tokens(total_text)
    threshold  = int(MAX_TOKENS * _PROMPT_BUDGET_FRACTION)

    if est_tokens > threshold:
        method = "tiktoken" if _TK_ENC else "heuristic"
        console.print(
            f"  [yellow]⚠  {filename}: spec + contracts ≈ {est_tokens:,} tokens "
            f"({method}, threshold {threshold:,}).  "
            f"Complex files may exceed MAX_TOKENS={MAX_TOKENS} and get a truncated "
            f"response.  Consider splitting into smaller sub-modules or raising "
            f"MAX_TOKENS before running.[/yellow]"
        )


# Regex patterns for extracting names from API_CONTRACT lines like:
#   ClassName  or  function_name(  or  CONSTANT :
_CONTRACT_CLASS_RE = re.compile(r'^\s{0,4}([A-Z][A-Za-z0-9_]+)\s*(?:—|-|:)', re.MULTILINE)
_CONTRACT_FUNC_RE  = re.compile(r'^\s{0,4}([a-z_][a-z_0-9]*)\s*\(', re.MULTILINE)


def _warn_if_contract_mismatch(filename: str, spec: str, code: str) -> None:
    """
    Improvement 4 — Incremental structural verification.

    Parse the API_CONTRACT section of the spec and check that every name
    declared there appears somewhere in the generated code (as a top-level
    `def`, `class`, or assignment).  Emit a yellow warning for each missing
    name so the developer (or a subsequent Fixer pass) can correct it before
    downstream files try to import the missing symbol and fail.

    This is intentionally *lightweight* — it does not run the code, parse an
    AST, or inspect actual exports.  It only checks that the symbol name
    appears in the file, which is enough to catch the most common failure mode:
    the coder generating a partial implementation that omits a class or function
    that another file will import.
    """
    contract_section = _extract_api_contract(spec)
    if not contract_section:
        return

    # Collect all names declared in the contract
    declared_classes = set(_CONTRACT_CLASS_RE.findall(contract_section))
    declared_funcs   = set(_CONTRACT_FUNC_RE.findall(contract_section))
    # Filter out common false-positive words that aren't real symbols
    _skip = {"Returns", "Raises", "Note", "Notes", "Args", "Example", "Usage"}
    declared_classes -= _skip

    missing = []
    for name in sorted(declared_classes):
        if not re.search(rf'\bclass\s+{re.escape(name)}\b', code):
            missing.append(f"class {name}")
    for name in sorted(declared_funcs):
        if name.startswith("_"):
            continue   # private helpers are not part of the public contract
        if not re.search(rf'\bdef\s+{re.escape(name)}\b', code):
            missing.append(f"def {name}")

    if missing:
        names_str = ", ".join(missing)
        console.print(
            f"  [yellow]⚠  {filename}: API_CONTRACT declares [{names_str}] "
            f"but these names were not found in the generated code.  "
            f"Downstream imports may fail.[/yellow]"
        )

# ── Prompts ───────────────────────────────────────────────────────────────────

ARCHITECT_SYSTEM = """You are an Autonomous Systems Engineer.
Your goal is to design a hardware-optimised software stack from a minimal brief.
You work in three strict phases. Do not skip or reorder them.

\u2501\u2501\u2501 PHASE 1 \u2014 MODEL SELECTION \u2501\u2501\u2501
You will be given hardware specs only \u2014 no models are pre-selected. You must choose
every model from scratch. Work through this in order:

1. INVENTORY the hardware: GPU model, VRAM total, CPU cores, RAM, OS, accelerator flags.

2. IDENTIFY the roles needed for the goal. For a vision/AI pipeline this typically means:
     - vision      \u2014 multimodal VLM for image understanding and reasoning
     - embedding   \u2014 vision embedding model for similarity search / identity matching
     - coding      \u2014 code generation model for dynamic analysis scripts
   Add or remove roles based on what the goal actually requires.

3. SELECT a specific model for each role **from the APPROVED MODELS list below**.
     These models are verified to be openly downloadable without HuggingFace gating.
     Do NOT select models outside this list — they will be rejected by validation.
     For each choice state:
     - Exact model identifier (must match an entry in APPROVED MODELS)
     - VRAM required at the chosen quantisation/precision
     - Why this model over alternatives (capability, VRAM fit, ROCm support, licence)
     - Runtime: vLLM in ROCm Docker (GPU) or Ollama on CPU host
     - Port assignment (suggest 8000, 8001, 8002... or 11434 for Ollama)

   APPROVED MODELS:
""" + _format_approved_models_for_prompt() + """

4. VERIFY the budget: list every model + VRAM, sum them, confirm the total fits
   within available VRAM with at least 15-20% headroom (e.g. ~2 GB on a 12 GB
   card, ~4 GB on a 24 GB card). A percentage scales correctly across card sizes
   and accounts for ROCm/vLLM driver overhead and context fragmentation on
   multi-GPU PCIe bifurcation setups. If it does not fit, swap to a smaller
   quantisation or different model and re-verify.

5. PRODUCE startup commands for each model:
     - GPU models: full `docker run` with all ROCm flags, port mapping, HF_TOKEN,
       --model, --max-model-len, and any tool-use flags needed.
     - CPU models: `ollama pull <tag>` then confirm port.
     - Multi-GPU: if multiple GPUs are detected, assign models to specific devices
       using `HIP_VISIBLE_DEVICES` (ROCm) or `CUDA_VISIBLE_DEVICES` (CUDA) in the
       startup commands to balance load across the PCIe bus. For example, assign
       the vision model to device 0 and the coding model to device 1 so each GPU
       handles its own PCIe lane under an x8/x8 bifurcation configuration.
   These commands will be stored in sys_config.json and printed at the end.

Do NOT proceed to Phase 2 until every role has a model, every GB is accounted for,
and every startup command is written out.

\u2501\u2501\u2501 PHASE 2 \u2014 SYSTEM CONFIG \u2501\u2501\u2501
Call write_config ONCE. This locks the model selection before any code is specified.
The config becomes the shared truth all generated files read at runtime.
It must include: model endpoints, model names, port numbers, storage paths,
VRAM allocations, ROCm/AMD flags, and the startup_commands for each model.
Do NOT call write_file before write_config has been called.

━━━ PHASE 3 — FILE SPECIFICATION ━━━
Call write_file once per component (max 8 files, not counting sys_config.json).
Each spec MUST contain all of these labelled sections:
  PURPOSE:        One sentence on what this file does.
  IMPORTS:        Every third-party and internal import, with exact module paths.
  CLASSES:        Each class, its __init__ signature, all public methods with signatures and return types.
  FUNCTIONS:      Each module-level function: signature, args, return type, logic summary.
  CONSTANTS:      Module-level config — load via `from config import cfg`, never hardcode values
                  and never open sys_config.json directly. Use cfg["key"] for all endpoints,
                  paths, model names, and thresholds.
  ERROR_HANDLING: What exceptions to catch and how (log, retry, raise, skip).
  API_CONTRACT:   Exact names and types this file exports for other files to import.
Specs must be mutually consistent. If file A imports class Foo from file B,
file B's spec must define class Foo with that exact name and interface.
Vague specs cause broken imports. Be precise.

━━━ DEPENDENCY MANIFEST RULE ━━━
After specifying all component files, call write_file once more for
`requirements.txt`. Its spec must list every third-party package (one per line,
with a minimum version pin, e.g. `anthropic>=0.25.0`) that appears in any IMPORTS
section across all files. Do NOT include stdlib modules. Include `pytest` as the
last entry so the Verifier can always run tests. Example format:
  anthropic>=0.25.0
  chromadb>=0.4.0
  fastapi>=0.110.0
  uvicorn>=0.29.0
  pytest>=8.0.0
The Verifier will run `pip install -r requirements.txt` before the import and
pytest stages, so missing third-party packages will be installed automatically.

━━━ FTP WATCHER RULE ━━━
The FTP folder is a DROP TARGET — write a watcher, not a server.
Files appear before transfers complete. The watcher must:
  - Poll file size every 2 seconds
  - Only proceed when size is unchanged for two consecutive polls
  - Confirm the file is not held open by another process using a portable
    exclusive-open check via pathlib / os (e.g. attempt to open the file with
    os.O_RDONLY | os.O_EXCL, or try-open and catch OSError). Do NOT use lsof
    as it may not be present in minimal Docker / container environments.

When all files are specified, call finalize."""

CODER_SYSTEM = """You are a specialist Python code generator.

You receive a filename, a detailed spec, and two context blocks:
  1. sys_config.json  — shown for reference so you understand what keys exist
  2. Prior file contracts — the API_CONTRACT of each file already generated,
     including config.py which is ALWAYS available

Return ONLY the complete file contents. No explanation. No markdown fences. No preamble.

Standards:
- Python 3.11+, Ubuntu 24.04
- Production quality — handle errors gracefully, log clearly
- Brief inline comments for non-obvious logic
- All imports at the top
- ALWAYS load runtime config with `from config import cfg` — never open or
  parse sys_config.json directly, never hardcode any endpoint, path, or model name
- Use the exact class names, function signatures, and import paths from the prior contracts
- For file-completion checks (e.g. FTP watcher): use a portable exclusive-open approach
  via pathlib / os (try-open with os.O_RDONLY | os.O_EXCL, or catch OSError). Do NOT
  use lsof — it is not available in all Linux container environments."""

TESTER_SYSTEM = """You are a specialist Python test writer.

You receive a filename, its spec, and its implementation code.
Write a pytest test file for it. Return ONLY the test file contents.
No explanation. No markdown fences. No preamble.

Rules:
- Use pytest. Import the module under test using its filename stem.
- Mock all external calls: HTTP endpoints, SQLite, ChromaDB, filesystem, subprocess.
  Use unittest.mock.patch and pytest fixtures. Tests must pass without any running services.
- Test each public function and class method from the API_CONTRACT section of the spec.
- Cover: happy path, empty/None inputs, and at least one error/exception path per function.
- Keep tests self-contained — no shared state between test functions.
- If the module loads sys_config.json at import time, patch builtins.open or
  use a tmp_path fixture to provide a minimal valid config."""

FIXER_SYSTEM = """You are a specialist Python code fixer.

You receive a filename, its original spec, its current (broken) code,
the type of failure (syntax / import / test), and the full error output.

Return ONLY the complete corrected file contents.
No explanation. No markdown fences. No preamble.

Rules:
- Fix only what the error describes. Do not refactor unrelated code.
- Preserve all function signatures and class names from the spec exactly —
  other files import from this one and will break if names change.
- If the error is an ImportError, check the import path against the provided
  prior contracts and sys_config.json, then correct it.
- If the error is a test failure, the test is correct — fix the implementation.
- All imports at the top. Use `from config import cfg` for all runtime values, never hardcode."""

# ── Phase 0: Feasibility Checker ──────────────────────────────────────────────

FEASIBILITY_SYSTEM = """You are a Hardware Feasibility Analyst.

You receive a project brief and must determine whether the stated goals are
achievable on the described hardware BEFORE any code is written.

Work through these steps in order:

1. PARSE the brief: extract the goal, the hardware spec (GPU, VRAM, CPU, RAM, OS),
   and any stated constraints (local-only, real-time, specific frameworks, etc.).

2. IDENTIFY the AI workload: what model roles are needed? Estimate VRAM for each
   (vision/VLM, embedding, detection, coding, etc.). Use realistic estimates for
   common quantisations (4-bit, 8-bit, fp16).

3. CHECK the VRAM budget: sum the required VRAM, add 15-20% headroom for OS and
   framework overhead (e.g. ~2 GB on a 12 GB card, ~4 GB on a 24 GB card — a
   flat 2 GB figure under-estimates on larger cards with ROCm/vLLM context
   fragmentation). Does it fit in the available VRAM? Note any tight margins.

4. CHECK CPU/RAM: are the non-GPU tasks (file watching, DB, API server) feasible
   on the stated CPU and RAM?

5. CHECK constraints: local-only, OS compatibility (ROCm, CUDA, Metal), storage,
   real-time requirements — flag anything that cannot be met.

6. VERDICT: call the feasibility_verdict tool with your findings.
   - If feasible: set status="feasible", describe what model stack fits.
   - If feasible with caveats: set status="feasible_with_caveats", list the caveats
     and what trade-offs (smaller models, reduced features) make it work.
   - If not feasible: set status="infeasible", explain exactly what is impossible
     and what hardware upgrade would be needed.

Be concrete. Quote VRAM numbers. Do not be vague."""

FEASIBILITY_TOOLS = [
    {
        "name": "feasibility_verdict",
        "description": "Record the feasibility assessment for the given brief and hardware.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["feasible", "feasible_with_caveats", "infeasible"],
                    "description": "Overall verdict."
                },
                "summary": {
                    "type": "string",
                    "description": "1-3 sentence plain-English summary of the verdict and its rationale."
                },
                "vram_estimate": {
                    "type": "string",
                    "description": (
                        "VRAM budget breakdown. Format: 'Role (model): Xgb  |  ...'  "
                        "Include total and available headroom."
                    )
                },
                "caveats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of caveats or trade-offs (empty if fully feasible)."
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hard blockers that make the goal infeasible (empty unless infeasible)."
                },
                "recommendations": {
                    "type": "string",
                    "description": (
                        "If infeasible: what hardware or scope changes would fix it. "
                        "If feasible/caveats: any optimisation tips."
                    )
                }
            },
            "required": ["status", "summary", "vram_estimate", "caveats", "blockers", "recommendations"]
        }
    }
]


@dataclass
class FeasibilityResult:
    status: str          # "feasible" | "feasible_with_caveats" | "infeasible"
    summary: str
    vram_estimate: str
    caveats: list
    blockers: list
    recommendations: str
    reasoning: str = ""


def run_feasibility_check(client: anthropic.Anthropic, brief: str) -> FeasibilityResult:
    """
    Phase 0: Ask the feasibility model whether the brief's goals can be met
    on the described hardware. Returns a FeasibilityResult.
    """
    messages = [{"role": "user", "content": f"Brief:\n\n{brief}"}]
    reasoning_parts: list[str] = []

    for _ in range(6):   # at most 6 turns to get the verdict
        response = client.messages.create(
            model=ARCHITECT_MODEL,
            max_tokens=MAX_TOKENS,
            system=FEASIBILITY_SYSTEM,
            tools=FEASIBILITY_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "text" and block.text.strip():
                reasoning_parts.append(block.text.strip())

            elif block.type == "tool_use" and block.name == "feasibility_verdict":
                inp = block.input
                result = FeasibilityResult(
                    status=inp["status"],
                    summary=inp["summary"],
                    vram_estimate=inp["vram_estimate"],
                    caveats=inp.get("caveats", []),
                    blockers=inp.get("blockers", []),
                    recommendations=inp.get("recommendations", ""),
                    reasoning="\n\n".join(reasoning_parts),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Verdict recorded.",
                })
                messages.append({"role": "user", "content": tool_results})
                return result

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    # Fallback if model never called the tool
    return FeasibilityResult(
        status="feasible_with_caveats",
        summary="Feasibility check did not produce a structured verdict. Proceeding with caution.",
        vram_estimate="Unknown",
        caveats=["Automated feasibility check inconclusive — review hardware manually."],
        blockers=[],
        recommendations="",
        reasoning="\n\n".join(reasoning_parts),
    )


def print_feasibility_result(result: FeasibilityResult):
    """Render the Phase 0 feasibility verdict panel."""
    status_styles = {
        "feasible":              ("bright_green", "✓  FEASIBLE"),
        "feasible_with_caveats": ("yellow",       "⚠  FEASIBLE WITH CAVEATS"),
        "infeasible":            ("bold red",     "✗  NOT FEASIBLE"),
    }
    color, label = status_styles.get(result.status, ("white", result.status.upper()))

    body_parts = [
        (f"{label}\n\n", color),
        ("SUMMARY\n", "bold grey70"),
        (wrap(result.summary) + "\n\n", "white"),
        ("VRAM ESTIMATE\n", "bold grey70"),
        (wrap(result.vram_estimate) + "\n", "white"),
    ]

    if result.caveats:
        body_parts.append(("\nCAVEATS\n", "bold yellow"))
        for c in result.caveats:
            body_parts.append((f"  • {c}\n", "yellow"))

    if result.blockers:
        body_parts.append(("\nBLOCKERS\n", "bold red"))
        for b in result.blockers:
            body_parts.append((f"  • {b}\n", "red"))

    if result.recommendations:
        body_parts.append(("\nRECOMMENDATIONS\n", "bold grey70"))
        body_parts.append((wrap(result.recommendations) + "\n", "grey74"))

    border = {
        "feasible":              "green",
        "feasible_with_caveats": "yellow",
        "infeasible":            "red",
    }.get(result.status, "grey30")

    console.print(Panel(
        Text.assemble(*body_parts),
        title=Text("  ⚙  Phase 0 — Feasibility Check  ", style="bold white"),
        title_align="left",
        border_style=border,
        padding=(0, 2),
    ))
    console.print()

    if result.reasoning:
        console.print(Padding(
            Text(wrap(result.reasoning), style=THEME["reasoning"], overflow="fold"),
            (0, 4)
        ))
        console.print()


ARCHITECT_TOOLS = [
    {
        "name": "find_free_port",
        "description": (
            "Query the host for a free TCP port in the range 8000-9000. "
            "Call this when a preferred port is already in use to obtain a "
            "conflict-free alternative before writing the config."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "integer",
                    "description": "Start of scan range (inclusive). Default 8000.",
                    "default": 8000
                },
                "end": {
                    "type": "integer",
                    "description": "End of scan range (exclusive). Default 9000.",
                    "default": 9000
                }
            },
            "required": []
        }
    },
    {
        "name": "write_config",
        "description": (
            "PHASE 2 — Call this ONCE before any write_file calls. "
            "Locks the hardware/model map. All generated code will read this file at runtime."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vram_allocation": {
                    "type": "string",
                    "description": (
                        "Full VRAM budget breakdown. Must account for every GB. "
                        "Format: 'ModelName (role): Xgb  |  ...'  Total must equal hardware VRAM."
                    )
                },
                "model_stack": {
                    "type": "object",
                    "description": (
                        "Mapping of role to model. Keys: vision, embedding, coding, (optional) detection. "
                        "Each value: {model_id, endpoint, port, runtime, vram_gb, rocm_flags (if any)}"
                    )
                },
                "startup_commands": {
                    "type": "object",
                    "description": (
                        "Startup command for each model role. Keys match model_stack keys. "
                        "Each value is a dict with: "
                        "'command' (the full shell command string to start the model), "
                        "'notes' (any one-time setup steps like huggingface-cli login or "
                        "ollama pull that must run before the command)."
                    )
                },
                "config_json": {
                    "type": "string",
                    "description": (
                        "Complete JSON content for sys_config.json. Must include: "
                        "model endpoints, model names, ports, storage paths (db, chroma, uploads, logs), "
                        "VRAM allocations, ROCm/AMD flags, and any thresholds (similarity, confidence, etc.). "
                        "This is the single source of truth all generated files will import."
                    )
                }
            },
            "required": ["vram_allocation", "model_stack", "startup_commands", "config_json"]
        }
    },
    {
        "name": "write_file",
        "description": (
            "PHASE 3 — Specify a component file. "
            "write_config MUST have been called first. Max 8 files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Relative path, e.g. watcher.py or tools/vision.py"
                },
                "spec": {
                    "type": "string",
                    "description": (
                        "Complete specification. MUST contain all labelled sections:\n"
                        "  PURPOSE, IMPORTS, CLASSES, FUNCTIONS, CONSTANTS, "
                        "ERROR_HANDLING, API_CONTRACT.\n"
                        "All endpoints/paths must reference sys_config.json keys, not hardcoded values.\n"
                        "Vague specs are not acceptable."
                    )
                }
            },
            "required": ["filename", "spec"]
        }
    },
    {
        "name": "finalize",
        "description": "Signal all files have been specified. Provide summary and run instructions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Architecture summary: what was built and why structured this way."
                },
                "run_instructions": {
                    "type": "string",
                    "description": "Step-by-step: install deps, configure, start the system."
                },
                "file_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All files in dependency order, starting with sys_config.json."
                }
            },
            "required": ["summary", "run_instructions", "file_list"]
        }
    }
]

DEFAULT_BRIEF = None   # No built-in default — user must supply a brief

# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 88) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else line
        for line in text.splitlines()
    )

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ── Hardware telemetry (Change 2: dynamic VRAM sensing) ──────────────────────

def get_gpu_info() -> str:
    """
    Query real-time GPU memory information and inject it into the Architect prompt.
    Prevents "hallucinated hardware" where the LLM assumes a card size that differs
    from the actual installed GPU.

    Tries ROCm (AMD) first, then falls back to NVIDIA, then returns a clear
    "no GPU detected" message so the Architect can still produce a CPU-only plan.
    """
    # ROCm / AMD path
    try:
        res = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            return f"[ROCm/AMD]\n{res.stdout.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # NVIDIA / CUDA path
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            return f"[NVIDIA/CUDA]\n{res.stdout.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "No GPU detected via CLI (rocm-smi / nvidia-smi not found or returned no data)."


# ── Port conflict detection (Change 4 / Improvement 1: robust port sensing) ───

# Sentinel values returned by _probe_port to distinguish states.
_PORT_OPEN       = "open"       # something is actively listening
_PORT_CLOSED     = "closed"     # nothing listening, safe to bind
_PORT_FILTERED   = "filtered"   # firewall / timeout — cannot determine state
_PORT_PRIVILEGED = "privileged" # port < 1024, probe refused by OS


def _probe_port(port: int, timeout: float = 1.0) -> str:
    """
    Attempt a non-blocking TCP connect to localhost:<port>.

    Returns one of the _PORT_* sentinel strings:
      - _PORT_OPEN       : connect succeeded → something is bound there
      - _PORT_CLOSED     : connection refused → port is free
      - _PORT_FILTERED   : timed out or no route → state unknown
      - _PORT_PRIVILEGED : OS refused the probe (port < 1024, no CAP_NET_BIND)

    Using a 1 s timeout (vs the old 0.5 s) reduces false negatives under
    moderate system load without adding noticeable latency for a handful
    of port checks.
    """
    if port < 1024:
        # Ports below 1024 require elevated privileges to bind; even probing
        # them can raise PermissionError on hardened systems.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                rc = s.connect_ex(("localhost", port))
                if rc == 0:
                    return _PORT_OPEN
                # ECONNREFUSED (111 on Linux) → nothing listening
                if rc in (111, 61):   # 61 = macOS ECONNREFUSED
                    return _PORT_CLOSED
                return _PORT_FILTERED
        except PermissionError:
            return _PORT_PRIVILEGED
        except OSError:
            return _PORT_FILTERED

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            rc = s.connect_ex(("localhost", port))
            if rc == 0:
                return _PORT_OPEN
            if rc in (111, 61):
                return _PORT_CLOSED
            return _PORT_FILTERED
    except OSError:
        return _PORT_FILTERED


def is_port_in_use(port: int) -> bool:
    """
    Backward-compatible wrapper: return True only when the port is confirmed
    open.  Filtered / privileged / closed states all return False (i.e. do not
    falsely block the port from being assigned).
    """
    return _probe_port(port) == _PORT_OPEN


def find_free_port(start: int = 8000, end: int = 9000) -> int | None:
    """
    Scan [start, end) and return the first port confirmed closed (free).
    Returns None if no free port is found in the range.

    Used by the Architect tool so it can auto-assign a conflict-free port
    rather than just warning the user.
    """
    for port in range(start, end):
        state = _probe_port(port)
        if state == _PORT_CLOSED:
            return port
    return None


def check_ports_in_model_stack(model_stack: dict) -> list[str]:
    """
    Scan every port in the model_stack for conflicts.

    Improvements (Improvement 1):
      - Uses _probe_port for precise state detection (open / closed / filtered /
        privileged) rather than a bare connect_ex.
      - Suggests a free alternative port (via find_free_port) when a conflict is
        detected, so the user has an immediately actionable fix.
      - Reports filtered ports as an advisory rather than a hard error so
        firewall-managed ports do not produce false positives.

    Returns a list of human-readable warning strings (empty list = all clear).
    """
    warnings: list[str] = []
    seen: dict[int, str] = {}
    for role, info in model_stack.items():
        if not isinstance(info, dict):
            continue
        raw_port = info.get("port")
        if raw_port is None:
            continue
        try:
            port = int(raw_port)
        except (ValueError, TypeError):
            continue

        # Duplicate port within the same stack
        if port in seen:
            alt = find_free_port()
            alt_hint = f"  Suggested free port: {alt}." if alt else ""
            warnings.append(
                f"Port conflict: both '{seen[port]}' and '{role}' are assigned "
                f"port {port}.{alt_hint}"
            )
        else:
            seen[port] = role

        state = _probe_port(port)
        if state == _PORT_OPEN:
            alt = find_free_port()
            alt_hint = f"  Suggested free port: {alt}." if alt else ""
            warnings.append(
                f"Port {port} (role '{role}') is already in use on localhost — "
                f"the container will fail to bind.{alt_hint}"
            )
        elif state == _PORT_FILTERED:
            warnings.append(
                f"Port {port} (role '{role}') appears filtered (no response "
                "within timeout). This may be a firewall rule — verify manually."
            )
        elif state == _PORT_PRIVILEGED:
            warnings.append(
                f"Port {port} (role '{role}') is below 1024 and requires "
                "elevated privileges to bind. Verify the runtime has CAP_NET_BIND_SERVICE."
            )
        # _PORT_CLOSED → no warning, port is free

    return warnings


def write_file(out_dir: Path, filename: str, code: str):
    filepath = out_dir / filename
    ensure_dir(filepath.parent)
    filepath.write_text(code, encoding="utf-8")
    return filepath


def install_requirements(out_dir: Path) -> bool:
    """
    Improvement 5 — Dependency Management.

    If a `requirements.txt` exists in out_dir, run:
        pip install -r requirements.txt --quiet

    in a subprocess so all third-party packages named in the Architect's
    manifest are available before the import and pytest verification stages.

    Returns True on success, False if pip exited non-zero (warning only —
    the build continues because some installs may fail in air-gapped envs).
    """
    req_path = out_dir / "requirements.txt"
    if not req_path.exists():
        return True   # no manifest yet — nothing to do

    console.print(
        f"  [{THEME['dim']}]📦  Installing dependencies from requirements.txt…[/{THEME['dim']}]"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        console.print(
            f"  [{THEME['success']}]✓[/{THEME['success']}]  "
            f"[{THEME['dim']}]requirements.txt installed successfully[/{THEME['dim']}]"
        )
        return True
    else:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        console.print(
            f"  [yellow]⚠  pip install returned non-zero exit code.  "
            f"Some packages may be missing.\n{stderr_tail}[/yellow]"
        )
        return False

def lang_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {"py": "python", "sh": "bash", "toml": "toml",
            "yaml": "yaml", "yml": "yaml", "md": "markdown",
            "json": "json", "txt": "text"}.get(ext.lstrip("."), "text")


# ── Checkpointing ─────────────────────────────────────────────────────────────

CHECKPOINT_FILENAME = "build_checkpoint.json"


def save_checkpoint(out_dir: Path, state: dict) -> None:
    """
    Persist full build state to disk so a crashed or interrupted run can resume.

    Improvement 2 — Atomic write pattern:
      Write to a sibling .tmp file first, then use os.replace() to atomically
      rename it over the target.  os.replace() is POSIX-atomic (single syscall)
      so a crash or power failure during the write can never leave a half-written
      checkpoint on disk — the previous valid checkpoint survives intact.

    Saved fields:
      - sys_config_json    : locked hardware config (str)
      - startup_commands   : model launch map (dict)
      - model_stack_data   : per-role model info (dict)
      - prior_specs        : accumulated API contracts (list)
      - completed_files    : filenames already written + verified (list)
      - best_pass_rate     : fraction of files that passed all checks (float)
      - iteration          : architect loop iteration when checkpoint was written (int)
      - timestamp          : ISO timestamp of the save (str)
    """
    checkpoint = {**state, "timestamp": datetime.now().isoformat()}
    target  = out_dir / CHECKPOINT_FILENAME
    tmp     = out_dir / (CHECKPOINT_FILENAME + ".tmp")
    try:
        tmp.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        os.replace(tmp, target)   # atomic on POSIX; near-atomic on Windows
        _build_logger.info(
            "CHECKPOINT saved  iteration=%s  completed=%d  pass_rate=%.2f",
            state.get("iteration", "?"),
            len(state.get("completed_files", [])),
            state.get("best_pass_rate", 0.0),
        )
    except OSError as exc:
        console.print(f"  [yellow]⚠  checkpoint save failed:[/yellow] {exc}")
        _build_logger.warning("CHECKPOINT save failed: %s", exc)
        # Best-effort cleanup of the temp file if it was left behind
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_checkpoint(out_dir: Path) -> dict | None:
    """
    Load a previous checkpoint from out_dir.
    Returns the parsed dict, or None if no checkpoint exists or it is corrupt.
    """
    path = out_dir / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be a JSON object")
        # Validate required keys are present
        required = {"sys_config_json", "prior_specs", "completed_files", "iteration"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"checkpoint missing keys: {missing}")
        return data
    except Exception as exc:
        console.print(f"  [yellow]⚠  checkpoint load failed (ignored):[/yellow] {exc}")
        return None


# ── Config loader — generated deterministically, not by the coder model ───────

# The API contract we inject as the first prior_spec so every coder call
# knows exactly what `from config import cfg` gives them.
CONFIG_PY_CONTRACT = """\
API_CONTRACT:
  cfg : dict  — the fully validated contents of sys_config.json, loaded once
                at import time from the directory containing config.py.
                Access any key directly: cfg["models"]["vision"]["endpoint"]
  reload_cfg() -> dict  — re-reads sys_config.json from disk and returns it,
                           useful for tests that patch the file."""


def generate_config_py(sys_config_json: str) -> str:
    """
    Build config.py deterministically from the known JSON content.
    Resolves sys_config.json relative to __file__ so it works regardless
    of the working directory the user runs from.
    Validates that the JSON is well-formed and raises a clear error if not.
    """
    # Extract top-level keys to include in the docstring for discoverability
    try:
        parsed = json.loads(sys_config_json)
        top_keys = ", ".join(f'"{k}"' for k in parsed.keys())
    except json.JSONDecodeError:
        top_keys = "(parse error — check sys_config.json)"

    return f'''\
"""
config.py — single source of truth for all runtime configuration.

Loads sys_config.json from the same directory as this file.
Import with:  from config import cfg

Top-level keys: {top_keys}
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "sys_config.json"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"sys_config.json not found at {{_CONFIG_PATH}}\\n"
            "Ensure sys_config.json is in the same directory as config.py."
        )
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sys_config.json is not valid JSON: {{exc}}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"sys_config.json must be a JSON object, got {{type(data).__name__}}")
    return data


def reload_cfg() -> dict:
    """Re-read sys_config.json from disk. Useful in tests."""
    return _load()


# Module-level singleton — loaded once on first import
cfg: dict = _load()
'''

# ── Model launcher ────────────────────────────────────────────────────────────

MODEL_LAUNCH_TIMEOUT = 180   # seconds to wait for a model endpoint to respond
MODEL_POLL_INTERVAL  = 5     # seconds between health-check polls


def _health_url(model_info: dict) -> str | None:
    """
    Derive a health-check URL from a model_stack entry.

    Priority order for endpoint selection (Improvement 3):
      1. Explicit `health_endpoint` key in the model_info dict (custom servers).
      2. `/health` — common in custom/FastAPI inference servers (llama.cpp, TGI…).
      3. `/ping`   — used by some lightweight serving frameworks.
      4. `/v1/models` — standard OpenAI-compatible (vLLM, LiteLLM).
      5. `/api/tags`  — Ollama native API.

    The function returns a *single* URL.  It picks the most specific endpoint
    it can infer; the caller (`wait_for_ready`) interprets any 200 as "ready".
    Returns None if no endpoint can be determined.
    """
    endpoint = model_info.get("endpoint") or model_info.get("url", "")
    runtime  = str(model_info.get("runtime", "")).lower()

    if not endpoint:
        port = model_info.get("port")
        if port:
            endpoint = f"http://localhost:{port}"
        else:
            return None

    endpoint = endpoint.rstrip("/")
    # Strip trailing /v1 so we work with the base URL throughout
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]

    # 1. Honour an explicit override (useful for custom inference servers)
    explicit = model_info.get("health_endpoint", "").strip()
    if explicit:
        return explicit if explicit.startswith("http") else f"{endpoint}{explicit}"

    # 2. Custom inference servers that advertise /health
    custom_runtime_keywords = ("tgi", "llama.cpp", "llamacpp", "fastapi", "custom", "triton")
    if any(kw in runtime for kw in custom_runtime_keywords):
        return f"{endpoint}/health"

    # 3. /ping (lightweight frameworks — e.g. bentoml, ray serve)
    if "ping" in runtime or "bento" in runtime or "ray" in runtime:
        return f"{endpoint}/ping"

    # 4. Ollama: /api/tags
    if "ollama" in runtime:
        return f"{endpoint}/api/tags"

    # 5. Default: OpenAI-compatible /v1/models (vLLM, LiteLLM, OpenAI proxy…)
    return f"{endpoint}/v1/models"


def wait_for_ready(role: str, model_info: dict) -> bool:
    """
    Poll the model's health endpoint until it returns 200 or timeout expires.

    Improvement 3 — Exponential backoff:
      Instead of a fixed MODEL_POLL_INTERVAL sleep between every attempt, the
      wait uses exponential back-off starting at 2 s and capping at
      MODEL_POLL_INTERVAL (default 5 s).  This reduces unnecessary waiting
      during the first few seconds when a model loads quickly, while still
      being patient for slow GPU weight loading.

      Back-off schedule (seconds): 2, 4, 5, 5, 5, …

    Returns True if ready, False on timeout.
    """
    url = _health_url(model_info)
    if not url:
        console.print(f"  [yellow]\u26a0[/yellow]  {role}: no endpoint to poll \u2014 assuming ready")
        return True

    deadline  = time.time() + MODEL_LAUNCH_TIMEOUT
    attempt   = 0
    base_wait = 2.0     # first back-off interval in seconds

    while time.time() < deadline:
        attempt += 1
        elapsed = int(time.time() - (deadline - MODEL_LAUNCH_TIMEOUT))
        try:
            with urllib.request.urlopen(url, timeout=4) as resp:
                if resp.status == 200:
                    console.print(
                        f"  [{THEME['success']}]\u2713[/{THEME['success']}]  "
                        f"[cyan]{role}[/cyan]  ready "
                        f"[{THEME['dim']}]({elapsed}s, {url})[/{THEME['dim']}]"
                    )
                    return True
        except urllib.error.HTTPError as exc:
            # 4xx/5xx from the server means it IS up, just returned an error
            # (e.g. 404 on a wrong path). Treat as "not ready yet" and keep polling.
            console.print(
                f"  [{THEME['dim']}]  {role}: HTTP {exc.code} from {url} "
                f"({elapsed}s) — still waiting…[/{THEME['dim']}]"
            )
        except Exception:
            pass

        # Exponential back-off capped at MODEL_POLL_INTERVAL
        wait = min(base_wait * (2 ** (attempt - 1)), float(MODEL_POLL_INTERVAL))
        console.print(
            f"  [{THEME['dim']}]  {role}: not ready yet ({elapsed}s elapsed) "
            f"— next check in {wait:.0f}s[/{THEME['dim']}]"
        )
        time.sleep(wait)

    console.print(
        f"  [red]\u2717[/red]  [cyan]{role}[/cyan]  "
        f"timed out after {MODEL_LAUNCH_TIMEOUT}s \u2014 check logs"
    )
    return False


def _extract_container_name(command: str) -> str | None:
    """
    Parse the --name <value> argument from a docker run command.
    Returns the container name string, or None if not present.
    """
    m = re.search(r'(?:^|\s)--name\s+(\S+)', command)
    return m.group(1) if m else None


def cleanup_container(name: str) -> None:
    """
    Force-remove a named Docker container if it exists.
    Called before launching a new container of the same name (Change 1) and
    registered with atexit so stale containers are cleaned up on normal exit.
    Suppresses all output — the container may not exist, which is fine.
    """
    subprocess.run(
        f"docker rm -f {name}",
        shell=True, capture_output=True,
    )


# Registry of container names launched this session so atexit can clean them up.
_launched_container_names: list[str] = []


def _atexit_cleanup_containers() -> None:
    """atexit handler: remove all containers launched by this build session."""
    for name in _launched_container_names:
        cleanup_container(name)


atexit.register(_atexit_cleanup_containers)


# Allowlist of safe command prefixes (Change 5: shell-injection guard)
_SAFE_CMD_PREFIXES = ("docker run", "ollama run", "ollama pull", "python", "python3")
# Characters that would allow chaining a second, attacker-controlled command
_CHAIN_CHARS = (";", "&&", "||", "`", "$(")


def _validate_command(command: str) -> str | None:
    """
    Validate that *command* is safe to pass to shell=True.

    Returns None if the command is acceptable, or a human-readable error string
    describing the problem if it should be refused.

    Rules (Change 5):
      1. Must start with a known-safe prefix.
      2. Must not contain shell-chaining operators that could smuggle a second
         command after a pipe or semicolon.

    Note: pipes (|) are *intentionally* permitted because legitimate Docker
    invocations sometimes pipe into `tee` for logging.  The check targets the
    more dangerous operators (;  &&  ||  `  $()) that are almost never needed
    in a model startup command.
    """
    stripped = command.strip()
    if not any(stripped.startswith(p) for p in _SAFE_CMD_PREFIXES):
        allowed = ", ".join(f"'{p}'" for p in _SAFE_CMD_PREFIXES)
        return (
            f"Command does not start with a known-safe prefix "
            f"(allowed: {allowed}):\n  {stripped[:120]}"
        )
    for ch in _CHAIN_CHARS:
        if ch in stripped:
            return (
                f"Command contains disallowed shell operator '{ch}' "
                f"which could allow command injection:\n  {stripped[:120]}"
            )
    return None


def launch_model(role: str, cmd_info: dict) -> subprocess.Popen | None:
    """
    Launch one model in the background (detached from this process).

    Changes vs. original:
      Change 1 — Docker zombie prevention:
        • Parse --name from the command.
        • Call cleanup_container(name) before starting, so a leftover container
          from a previous crashed run never blocks the new launch.
        • Register the name in _launched_container_names for atexit cleanup.
      Change 5 — Shell-injection guard:
        • Validate the command string before executing it.
        • Refuse commands that don't start with a safe prefix or that contain
          shell-chaining operators.

    Returns the Popen handle, or None if no command was provided or the command
    failed validation.
    """
    command = cmd_info.get("command", "").strip()
    notes   = cmd_info.get("notes",   "").strip()

    if not command:
        console.print(f"  [yellow]⚠[/yellow]  {role}: no startup command — skipping")
        return None

    # Change 5: validate before executing
    validation_error = _validate_command(command)
    if validation_error:
        console.print(
            f"  [bold red]✗  {role}: command rejected by security policy[/bold red]\n"
            f"  [yellow]{validation_error}[/yellow]\n"
            f"  [grey42]Skipping launch — start this model manually.[/grey42]"
        )
        return None

    # Change 1: pre-launch container cleanup
    container_name = _extract_container_name(command)
    if container_name:
        cleanup_container(container_name)
        _launched_container_names.append(container_name)

    if notes:
        console.print(Padding(Text(f"# {notes}", style=THEME["dim"]), (0, 4)))
    console.print(Padding(Text(command, style="bright_white"), (0, 4)))

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,   # survive if build.py exits
        )
        console.print(
            f"  [{THEME['success']}]▶[/{THEME['success']}]  "
            f"[cyan]{role}[/cyan]  launched  "
            f"[{THEME['dim']}](pid {proc.pid})[/{THEME['dim']}]"
        )
        return proc
    except Exception as exc:
        console.print(f"  [red]✗[/red]  {role}: launch failed — {exc}")
        return None


def launch_all_models(model_stack: dict, startup_commands: dict) -> list[tuple[str, subprocess.Popen]]:
    """
    Launch each model sequentially. After each launch, poll its health endpoint
    and wait until it responds before starting the next one. This prevents VRAM
    contention when two GPU models try to load simultaneously.

    Returns list of (role, Popen) for all successfully launched processes.
    """
    print_phase("LAUNCHING MODELS  \u00b7  sequential \u2192 wait for ready \u2192 next", "bold magenta")

    launched: list[tuple[str, subprocess.Popen]] = []
    all_ready = True

    for role, cmd_info in startup_commands.items():
        if not isinstance(cmd_info, dict):
            cmd_info = {"command": str(cmd_info), "notes": ""}

        model_info = model_stack.get(role, {})
        console.print(
            f"\n  [bold magenta]{role.upper()}[/bold magenta]  "
            f"[{THEME['dim']}]{model_info.get('model_id', '')}[/{THEME['dim']}]"
        )

        proc = launch_model(role, cmd_info)
        if proc:
            launched.append((role, proc))
            if not wait_for_ready(role, model_info):
                all_ready = False
                console.print(
                    f"  [yellow]\u26a0  Continuing \u2014 "
                    f"{role} may still be loading weights[/yellow]"
                )

    console.print()
    if all_ready:
        console.print(Padding(Text("\u2713  All models ready.", style="bold bright_green"), (0, 4)))
    else:
        console.print(Padding(
            Text(
                "\u26a0  Some models did not respond within the timeout.\n"
                "  They may still be downloading weights. "
                "Check with:  docker ps  |  ollama ps",
                style="yellow",
            ),
            (0, 4)
        ))
    console.print()
    return launched


# ── Phase 1: Architect ────────────────────────────────────────────────────────

# ── Verification ─────────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    passed: bool
    stage: str                    # "syntax" | "import" | "pytest" | "ok"
    error: str = ""
    test_code: str = ""           # populated after test-writer runs
    test_path: Path | None = None


def check_syntax(code: str, filename: str) -> VerifyResult:
    """Stage 1: ast.parse — instant, zero deps."""
    try:
        ast.parse(code)
        return VerifyResult(passed=True, stage="syntax")
    except SyntaxError as e:
        return VerifyResult(
            passed=False, stage="syntax",
            error=f"SyntaxError in {filename} line {e.lineno}: {e.msg}\n  {e.text}"
        )


def check_imports(filepath: Path, out_dir: Path) -> VerifyResult:
    """Stage 2: import the module in a subprocess with out_dir on sys.path.

    For files in subdirectories (e.g. tools/frame_extractor.py) we also add
    filepath.parent to sys.path so that ``import frame_extractor`` resolves
    correctly while imports like ``from tools.vision import VisionTool`` still
    resolve via out_dir.
    """
    module = filepath.stem
    file_dir = str(filepath.parent)
    pythonpath = f"{out_dir}:{file_dir}"
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{file_dir}'); sys.path.insert(0, '{out_dir}'); import {module}",
        ],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    if result.returncode == 0:
        return VerifyResult(passed=True, stage="import")
    # Filter to the most useful part of the traceback
    stderr = result.stderr.strip()
    last_lines = "\n".join(stderr.splitlines()[-6:])
    return VerifyResult(passed=False, stage="import", error=last_lines)


def generate_tests(client: anthropic.Anthropic, filename: str, spec: str,
                   code: str, sys_config_json: str) -> str:
    """Ask the tester model to write a pytest file for this module."""
    _build_logger.info("TESTER  generating tests for  %s", filename)
    user_content = (
        f"sys_config.json:\n{sys_config_json}\n\n"
        f"File: {filename}\n\nSpec:\n{spec}\n\nImplementation:\n{code}"
    )
    response = client.messages.create(
        model=TESTER_MODEL,
        max_tokens=MAX_TOKENS,
        system=TESTER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    result = next((b.text for b in response.content if b.type == "text"), "")
    _log_llm_call("generate_tests", TESTER_MODEL, TESTER_SYSTEM, user_content, result)
    return result


def check_pytest(test_path: Path, out_dir: Path) -> VerifyResult:
    """Stage 3: run pytest on the generated test file."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-x", "--tb=short", "-q"],
        capture_output=True, text=True, timeout=120,
        cwd=str(out_dir),
        env={**os.environ, "PYTHONPATH": str(out_dir)},
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return VerifyResult(passed=True, stage="pytest", test_path=test_path)
    # Keep the last 30 lines — enough to see failures without flooding the terminal
    trimmed = "\n".join(output.splitlines()[-30:])
    return VerifyResult(passed=False, stage="pytest", error=trimmed, test_path=test_path)


def _get_dir_snapshot(out_dir: Path) -> str:
    """
    Return a compact listing of every .py file currently present in out_dir.
    Injected into fixer prompts when an ImportError occurs so the model can
    see what files actually exist on disk vs. what the spec assumed.
    """
    py_files = sorted(out_dir.rglob("*.py"))
    if not py_files:
        return "(no .py files found in output directory)"
    lines = [f"  {p.relative_to(out_dir)}" for p in py_files]
    return "\n".join(lines)


async def _fix_file_async(filename: str, spec: str,
                          broken_code: str, failure: VerifyResult,
                          prior_specs: list[dict], sys_config_json: str,
                          out_dir: Path) -> str:
    """Use Claude Code Agent SDK to fix a broken file.

    The agent can read neighbouring files, run commands, and inspect errors
    autonomously — a significant upgrade over the single-shot API approach.
    """
    contracts = ""
    if prior_specs:
        contracts = "Prior file contracts (exact import targets):\n" + "\n".join(
            f"  {p['filename']}:\n{p['api_contract']}" for p in prior_specs
        )

    _build_logger.info(
        "FIXER  %s  stage=%s  attempt (via Claude Code Agent SDK)", filename, failure.stage
    )
    _build_logger.debug("FIXER error detail:\n%s", failure.error[:2000])

    target = out_dir / filename
    prompt = (
        f"{FIXER_SYSTEM}\n\n"
        f"sys_config.json:\n{sys_config_json}\n\n"
        f"{contracts}\n\n"
        f"File to fix: {filename} (at {target})\n\n"
        f"Original spec:\n{spec}\n\n"
        f"Current broken code:\n{broken_code}\n\n"
        f"Failure stage: {failure.stage}\n"
        f"Error output:\n{failure.error}\n\n"
        f"Read the file, inspect neighbouring files if needed, fix the issue, "
        f"and write the corrected version to {target}."
    )

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        system_prompt=FIXER_SYSTEM,
        cwd=str(out_dir),
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            pass  # we read from disk below

    # Read back the corrected file
    if target.exists():
        code = target.read_text(encoding="utf-8")
        _build_logger.info("FIXER  %s  result length=%d chars (agent)", filename, len(code))
        return code

    _build_logger.warning("FIXER  %s  agent did not write file — returning broken code", filename)
    return broken_code


def fix_file(client: anthropic.Anthropic, filename: str, spec: str,
             broken_code: str, failure: VerifyResult,
             prior_specs: list[dict], sys_config_json: str,
             out_dir: Path | None = None) -> str:
    """Fix a broken file. Uses Claude Code Agent SDK when out_dir is available,
    falls back to direct API call otherwise."""

    if out_dir is not None:
        return asyncio.run(
            _fix_file_async(filename, spec, broken_code, failure,
                            prior_specs, sys_config_json, out_dir)
        )

    # Fallback: direct API call (no output directory context)
    contracts = ""
    if prior_specs:
        contracts = "Prior file contracts (exact import targets):\n" + "\n".join(
            f"  {p['filename']}:\n{p['api_contract']}" for p in prior_specs
        )

    _build_logger.info(
        "FIXER  %s  stage=%s  attempt (fix call)", filename, failure.stage
    )
    _build_logger.debug("FIXER error detail:\n%s", failure.error[:2000])

    response = client.messages.create(
        model=CODER_MODEL,
        max_tokens=MAX_TOKENS,
        system=FIXER_SYSTEM,
        messages=[{"role": "user", "content": (
            f"sys_config.json:\n{sys_config_json}\n\n"
            f"{contracts}\n\n"
            f"File to fix: {filename}\n\n"
            f"Original spec:\n{spec}\n\n"
            f"Current broken code:\n{broken_code}\n\n"
            f"Failure stage: {failure.stage}\n"
            f"Error output:\n{failure.error}"
        )}],
    )
    result = next((b.text for b in response.content if b.type == "text"), broken_code)
    _build_logger.info("FIXER  %s  result length=%d chars", filename, len(result))
    return result


def get_retry_delay(attempt: int, base: float = 1.0, max_delay: float = 8.0) -> float:
    """
    Centralized retry back-off schedule (Change 2).
    Returns the number of seconds to wait before the given attempt.
    Uses exponential back-off capped at max_delay: 0, 1, 2, 4, 8, 8, …

    Keeping this in one place makes it easy to tune without hunting through the loop.
    """
    if attempt <= 1:
        return 0.0
    return min(base * (2 ** (attempt - 2)), max_delay)


def _code_is_empty(code: str) -> bool:
    """Return True if generated code is blank or a known failure sentinel."""
    stripped = code.strip()
    return not stripped or stripped == "# code generation failed"


def verify_and_fix(
    client: anthropic.Anthropic,
    filename: str,
    spec: str,
    initial_code: str,
    filepath: Path,
    out_dir: Path,
    prior_specs: list[dict],
    sys_config_json: str,
) -> tuple[str, list[VerifyResult]]:
    """
    Run all three verification stages on a generated file.
    On failure, call the fixer and retry up to MAX_RETRIES times.
    Returns (final_code, list_of_all_results).

    Retry strategy per stage:
      syntax  — send broken code + error to fixer
      import  — send broken code + error + prior contracts to fixer
      pytest  — send broken code + test output to fixer (test is correct)

    Improvements (Changes 2, 4, 5, 9):
      - Retry back-off via get_retry_delay() — centralised, easy to tune
      - Empty/sentinel code guard before each attempt
      - Stale result state cleared (reset) at the start of each retry pass
      - Standardised per-attempt log line: file | attempt | stage | pass/fail
      - NaN-equivalent guard: empty generated code aborts early with a clear error
    """
    code = initial_code
    all_results: list[VerifyResult] = []

    # Only .py files get verified
    if not filename.endswith(".py"):
        all_results.append(VerifyResult(passed=True, stage="ok"))
        return code, all_results

    for attempt in range(1, MAX_RETRIES + 1):
        # ── Change 6: empty-code recovery ────────────────────────────────────
        # Original behaviour: abort immediately on empty/sentinel code.
        # New behaviour: on attempt 1 the code was already verified non-empty
        # before we entered verify_and_fix, so this branch guards subsequent
        # retry iterations where fix_file itself returned empty (rare, but
        # possible on a token-limit hit or transient API refusal).
        # Instead of giving up, we call fix_file once more with an explicit
        # "you returned empty output" instruction so the system can self-heal.
        if _code_is_empty(code):
            if attempt < MAX_RETRIES:
                console.print(
                    f"  [yellow]⚠  {filename} | attempt {attempt} | "
                    f"fixer returned empty output — retrying with explicit prompt[/yellow]"
                )
                empty_failure = VerifyResult(
                    passed=False, stage="syntax",
                    error=(
                        "The previous attempt resulted in an empty file. "
                        "Please ensure you output the FULL implementation code "
                        "without truncation. Do not produce a placeholder or an "
                        "empty response."
                    ),
                )
                all_results.append(empty_failure)
                code = fix_file(
                    client, filename, spec,
                    "# (empty — previous generation produced no output)",
                    empty_failure, prior_specs, sys_config_json, out_dir,
                )
                filepath.write_text(code, encoding="utf-8")
                continue
            else:
                err = "code generation returned empty output after all retries — aborting"
                console.print(f"  [red]✗  {filename} | attempt {attempt} | empty code — giving up[/red]")
                all_results.append(VerifyResult(passed=False, stage="syntax", error=err))
                break

        # Apply back-off before retry attempts (not before the first pass)
        delay = get_retry_delay(attempt)
        if delay:
            time.sleep(delay)

        # ── Stage 1: syntax ───────────────────────────────────────────────────
        r = check_syntax(code, filename)
        all_results.append(r)
        _log_stage(filename, attempt, "syntax", r.passed)
        if not r.passed:
            if attempt < MAX_RETRIES:
                code = fix_file(client, filename, spec, code, r, prior_specs, sys_config_json, out_dir)
                filepath.write_text(code, encoding="utf-8")
            continue

        # ── Stage 2: imports ──────────────────────────────────────────────────
        r = check_imports(filepath, out_dir)
        all_results.append(r)
        _log_stage(filename, attempt, "import", r.passed)
        if not r.passed:
            if attempt < MAX_RETRIES:
                # out_dir passed so fixer can see which files exist on disk
                code = fix_file(client, filename, spec, code, r, prior_specs, sys_config_json, out_dir)
                filepath.write_text(code, encoding="utf-8")
            continue

        # ── Stage 3: generate tests then run pytest ───────────────────────────
        test_code = generate_tests(client, filename, spec, code, sys_config_json)
        test_filename = f"test_{Path(filename).stem}.py"
        test_path = out_dir / "tests" / test_filename
        ensure_dir(test_path.parent)
        test_path.write_text(test_code, encoding="utf-8")

        r = check_pytest(test_path, out_dir)
        r.test_code = test_code
        all_results.append(r)
        _log_stage(filename, attempt, "pytest", r.passed)
        if not r.passed:
            if attempt < MAX_RETRIES:
                # Fix the implementation, not the test
                code = fix_file(client, filename, spec, code, r, prior_specs, sys_config_json, out_dir)
                filepath.write_text(code, encoding="utf-8")
            continue

        # All three passed
        break

    return code, all_results


def _log_stage(filename: str, attempt: int, stage: str, passed: bool) -> None:
    """
    Standardised single-line log entry for each verification stage (Change 5).
    Format mirrors training-loop convention: File | Attempt N | Stage | PASS/FAIL
    """
    status = f"[bright_green]PASS[/bright_green]" if passed else f"[red]FAIL[/red]"
    console.print(
        f"  [{THEME['dim']}]file[/{THEME['dim']}] {filename}"
        f"  [{THEME['dim']}]attempt[/{THEME['dim']}] {attempt}"
        f"  [{THEME['dim']}]stage[/{THEME['dim']}] {stage}"
        f"  {status}"
    )
    # Improvement 6: mirror to the persistent log file
    level = logging.INFO if passed else logging.WARNING
    _build_logger.log(level, "VERIFY  %s  attempt=%d  stage=%s  %s",
                      filename, attempt, stage, "PASS" if passed else "FAIL")


def _extract_api_contract(spec: str) -> str:
    """Pull the API_CONTRACT section from a spec, or return a truncated fallback."""
    idx = spec.upper().find("API_CONTRACT")
    if idx != -1:
        snippet = spec[idx:]
        sections = list(re.finditer(r'\n[A-Z_]{3,}:', snippet))
        if len(sections) > 1:
            snippet = snippet[:sections[1].start()]
        return snippet.strip()
    return spec[:400].strip()


def run_architect(client: anthropic.Anthropic, brief: str):
    """
    Three-phase architect loop. Yields typed events.

    Phase enforcement:
      - write_file before write_config  → rejected with an error tool_result
      - write_config called twice        → second call ignored, warning yielded
      - finalize before write_config     → rejected

    Yielded event types:
      reasoning   — text block from the model
      hw_audit    — write_config accepted: {vram_allocation, model_stack, config_json}
      file_spec   — write_file accepted:   {filename, spec}
      finalize    — finalize called:        {summary, run_instructions, file_list}
      phase_error — enforcement rejection:  {message}
    """
    # Change 2: inject real-time GPU telemetry so the Architect reasons from
    # actual hardware, not hallucinated specs from the brief alone.
    gpu_info = get_gpu_info()
    messages = [{"role": "user", "content": (
        f"Brief:\n\n{brief}\n\n"
        f"--- LIVE HARDWARE TELEMETRY (authoritative — use this, not the brief's hardware "
        f"description, for all VRAM budget calculations) ---\n{gpu_info}"
    )}]
    config_locked = False
    iterations = 0

    while iterations < MAX_ARCH_ITERS:
        iterations += 1

        response = client.messages.create(
            model=ARCHITECT_MODEL,
            max_tokens=MAX_TOKENS,
            system=ARCHITECT_SYSTEM,
            tools=ARCHITECT_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "text" and block.text.strip():
                yield {"type": "reasoning", "text": block.text.strip()}

            elif block.type == "tool_use":

                # ── find_free_port ────────────────────────────────────────────
                if block.name == "find_free_port":
                    start = int(block.input.get("start", 8000))
                    end   = int(block.input.get("end",   9000))
                    free  = find_free_port(start, end)
                    if free is not None:
                        content = f"Free port found: {free}. Use this port in your config."
                    else:
                        content = (
                            f"No free port found in range [{start}, {end}). "
                            "Try a different range or free up a port manually."
                        )
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     content,
                    })

                # ── write_config ──────────────────────────────────────────────
                elif block.name == "write_config":
                    if config_locked:
                        msg = "write_config already called. Hardware map is locked. Proceed with write_file."
                        yield {"type": "phase_error", "message": msg}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": msg,
                            "is_error": True,
                        })
                    else:
                        # ── Validate model stack against approved registry ──
                        model_stack = block.input.get("model_stack", {})
                        model_errors = _validate_model_stack(model_stack)
                        if model_errors:
                            msg = (
                                "MODEL APPROVAL ERROR: One or more models are not in the "
                                "approved list. Gated models require prior HuggingFace "
                                "approval and will fail at download time.\n"
                                + "\n".join(model_errors)
                                + "\n\nRevise your model selections using only "
                                "APPROVED MODELS, then call write_config again."
                            )
                            yield {"type": "phase_error", "message": msg}
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": msg,
                                "is_error": True,
                            })
                        else:
                            config_locked = True
                            yield {
                                "type": "hw_audit",
                                "vram_allocation":  block.input["vram_allocation"],
                                "model_stack":      block.input["model_stack"],
                                "startup_commands": block.input.get("startup_commands", {}),
                                "config_json":      block.input["config_json"],
                            }
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "sys_config.json locked. Proceed to Phase 3: write_file calls.",
                            })

                # ── write_file ────────────────────────────────────────────────
                elif block.name == "write_file":
                    if not config_locked:
                        msg = (
                            "PHASE ERROR: write_file called before write_config. "
                            "You must complete Phase 2 (call write_config) before specifying files."
                        )
                        yield {"type": "phase_error", "message": msg}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": msg,
                            "is_error": True,
                        })
                    else:
                        yield {
                            "type": "file_spec",
                            "filename": block.input["filename"],
                            "spec":     block.input["spec"],
                        }
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Queued: {block.input['filename']}",
                        })

                # ── finalize ──────────────────────────────────────────────────
                elif block.name == "finalize":
                    if not config_locked:
                        msg = "PHASE ERROR: finalize called before write_config. Complete Phase 2 first."
                        yield {"type": "phase_error", "message": msg}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": msg,
                            "is_error": True,
                        })
                    else:
                        yield {"type": "finalize", **block.input}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Finalized.",
                        })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

# ── Phase 2: Coder ────────────────────────────────────────────────────────────

async def _generate_file_async(filename: str, spec: str,
                               prior_specs: list[dict], sys_config_json: str,
                               out_dir: Path) -> str:
    """Use Claude Code Agent SDK to generate a file."""
    parts = []

    parts.append(f"sys_config.json (read this at runtime via json.load — do not hardcode any values from it):\n{sys_config_json}")

    if prior_specs:
        parts.append("Prior files in this project (import from these, do not reimplement):")
        for p in prior_specs:
            parts.append(f"  File: {p['filename']}\n  API contract:\n{p['api_contract']}")

    parts.append(f"File to implement: {filename}\n\nSpecification:\n{spec}")

    user_content = "\n\n".join(parts)
    _build_logger.info("CODER  generating  %s  (via Claude Code Agent SDK)", filename)

    prompt = (
        f"{CODER_SYSTEM}\n\n"
        f"{user_content}\n\n"
        f"Write the complete file to: {out_dir / filename}"
    )

    options = ClaudeAgentOptions(
        allowed_tools=["Write", "Read", "Edit"],
        permission_mode="acceptEdits",
        system_prompt=CODER_SYSTEM,
        cwd=str(out_dir),
    )

    result_text = "# code generation failed"
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_text = message.result

    # Read back what the agent wrote to disk
    target = out_dir / filename
    if target.exists():
        code = target.read_text(encoding="utf-8")
        _log_llm_call("generate_file", "claude-code-agent", CODER_SYSTEM, user_content, code)
        return code

    # Fallback: if the agent returned code in its result text instead of writing
    _log_llm_call("generate_file", "claude-code-agent", CODER_SYSTEM, user_content, result_text)
    return result_text


def generate_file(client: anthropic.Anthropic, filename: str, spec: str,
                  prior_specs: list[dict], sys_config_json: str,
                  out_dir: Path | None = None) -> str:
    """Call Claude Code Agent SDK to generate a file. Falls back to API if out_dir is None."""
    _warn_if_spec_too_large(filename, spec, prior_specs)

    if out_dir is None:
        # Fallback to direct API call if no output directory provided
        parts = []
        parts.append(f"sys_config.json (read this at runtime via json.load — do not hardcode any values from it):\n{sys_config_json}")
        if prior_specs:
            parts.append("Prior files in this project (import from these, do not reimplement):")
            for p in prior_specs:
                parts.append(f"  File: {p['filename']}\n  API contract:\n{p['api_contract']}")
        parts.append(f"File to implement: {filename}\n\nSpecification:\n{spec}")
        user_content = "\n\n".join(parts)
        _build_logger.info("CODER  generating  %s", filename)
        response = client.messages.create(
            model=CODER_MODEL, max_tokens=MAX_TOKENS,
            system=CODER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        result = next((b.text for b in response.content if b.type == "text"), "# code generation failed")
        _log_llm_call("generate_file", CODER_MODEL, CODER_SYSTEM, user_content, result)
        return result

    return asyncio.run(_generate_file_async(filename, spec, prior_specs, sys_config_json, out_dir))

# ── Rich UI helpers ───────────────────────────────────────────────────────────

THEME = {
    "architect": "bold cyan",
    "coder":     "bold green",
    "filename":  "bright_green",
    "spec":      "grey58",
    "reasoning": "grey70",
    "rule":      "grey30",
    "dim":       "grey42",
    "accent":    "orange1",
    "success":   "bright_green",
    "header":    "bold white",
}

def print_hw_audit(vram_allocation: str, model_stack: dict,
                   startup_commands: dict, config_json: str):
    """Display the hardware audit / config lock panel."""
    # Model stack table
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2),
                  header_style="grey50")
    table.add_column("Role",     style="cyan",         no_wrap=True)
    table.add_column("Model",    style="bright_white",  no_wrap=True)
    table.add_column("Runtime",  style=THEME["dim"],    no_wrap=True)
    table.add_column("VRAM",     style="yellow",        no_wrap=True)
    table.add_column("Port",     style=THEME["dim"],    no_wrap=True)

    for role, info in model_stack.items():
        if isinstance(info, dict):
            table.add_row(
                role,
                str(info.get("model_id", "—")),
                str(info.get("runtime", "—")),
                str(info.get("vram_gb", "—")) + " GB",
                str(info.get("port", "—")),
            )
        else:
            table.add_row(role, str(info), "—", "—", "—")

    console.print(Panel(
        Text.assemble(
            ("VRAM ALLOCATION\n", "bold grey70"),
            (wrap(vram_allocation) + "\n\n", "white"),
        ),
        title=Text("  \u2699  Hardware Map Locked  ", style="bold yellow"),
        title_align="left",
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print(Padding(table, (0, 4)))
    console.print()

    # Startup commands — one panel per role
    if startup_commands:
        console.print(Padding(
            Text("STARTUP COMMANDS", style="bold grey70"),
            (0, 4)
        ))
        for role, info in startup_commands.items():
            if not isinstance(info, dict):
                info = {"command": str(info), "notes": ""}
            notes = info.get("notes", "")
            cmd   = info.get("command", "")
            body_parts = []
            if notes:
                body_parts.append(("# " + notes + "\n", "grey50"))
            body_parts.append((cmd, "bright_white"))
            console.print(Padding(
                Panel(
                    Text.assemble(*body_parts),
                    title=Text(f"  {role}  ", style="cyan"),
                    title_align="left",
                    border_style="grey30",
                    padding=(0, 2),
                ),
                (0, 4)
            ))
        console.print()

    # config JSON preview
    try:
        pretty = json.dumps(json.loads(config_json), indent=2)
    except Exception:
        pretty = config_json
    preview = "\n".join(pretty.splitlines()[:40])
    if len(pretty.splitlines()) > 40:
        preview += f"\n  \u2026 {len(pretty.splitlines()) - 40} more lines"
    console.print(Padding(
        Syntax(preview, "json", theme="monokai", line_numbers=False,
               background_color="default"),
        (0, 4, 1, 4)
    ))


def print_phase_error(message: str):
    console.print(Padding(
        Text(f"⚠  {message}", style="bold red"),
        (0, 4)
    ))
    console.print()


def print_header():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("⚡  AUTONOMOUS BUILDER\n", "bold white"),
            ("Phase 0: feasibility  ·  Architect: design  ·  Coder: generate  ·  Files written to disk", THEME["dim"]),
        ),
        border_style="grey30",
        padding=(0, 2),
    ))
    console.print()

def print_phase(label: str, color: str):
    console.print(Rule(f"[{color}]{label}[/{color}]", style=THEME["rule"]))
    console.print()

def print_reasoning(text: str):
    console.print(Padding(
        Text(wrap(text), style=THEME["reasoning"], overflow="fold"),
        (0, 4)
    ))
    console.print()

def print_file_spec(filename: str, spec: str):
    preview = wrap(spec[:300]) + ("…" if len(spec) > 300 else "")
    console.print(Panel(
        Text.assemble(
            (preview, THEME["spec"]),
        ),
        title=Text(f"  {filename}  ", style=THEME["filename"]),
        title_align="left",
        border_style="dark_green",
        padding=(0, 2),
    ))
    console.print()

def print_generating(filename: str):
    console.print(f"  [{THEME['coder']}]⟳[/{THEME['coder']}]  [{THEME['filename']}]{filename}[/{THEME['filename']}]  [{THEME['dim']}]generating…[/{THEME['dim']}]")

def print_file_written(filename: str, filepath: Path, lines: int):
    console.print(
        f"  [{THEME['success']}]✓[/{THEME['success']}]  "
        f"[{THEME['filename']}]{filename}[/{THEME['filename']}]  "
        f"[{THEME['dim']}]{lines} lines → {filepath}[/{THEME['dim']}]"
    )

def print_verify_result(results: list[VerifyResult], filename: str, attempt: int):
    """Render a compact verification status row."""
    # Build stage indicators
    stages = {"syntax": "?", "import": "?", "pytest": "?"}
    final_passed = True
    for r in results:
        if r.stage in stages:
            stages[r.stage] = "✓" if r.passed else "✗"
        if not r.passed:
            final_passed = False

    def badge(sym: str, label: str) -> str:
        color = "bright_green" if sym == "✓" else ("red" if sym == "✗" else "grey42")
        return f"[{color}]{sym} {label}[/{color}]"

    row = "  " + "  ".join(badge(v, k) for k, v in stages.items())
    if attempt > 1:
        row += f"  [yellow](attempt {attempt})[/yellow]"
    console.print(row)

    # Print error detail for the last failed stage
    last_fail = next((r for r in reversed(results) if not r.passed), None)
    if last_fail:
        console.print(Padding(
            Panel(
                Text(last_fail.error[:800], style="red"),
                title=Text(f"  {last_fail.stage} failure  ", style="bold red"),
                title_align="left",
                border_style="red",
                padding=(0, 2),
            ),
            (0, 4)
        ))
    console.print()


def print_verification_summary(build_results: list[dict]):
    """Final table: all files, their verification status, attempt count."""
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2),
                  header_style="grey50")
    table.add_column("File",     style="bright_white")
    table.add_column("Syntax",   justify="center")
    table.add_column("Import",   justify="center")
    table.add_column("Pytest",   justify="center")
    table.add_column("Attempts", justify="right", style="grey50")
    table.add_column("Status",   justify="center")

    def sym(results, stage):
        for r in results:
            if r.stage == stage:
                return "[bright_green]✓[/bright_green]" if r.passed else "[red]✗[/red]"
        return "[grey42]—[/grey42]"

    for br in build_results:
        results = br["results"]
        attempts = br["attempts"]
        passed = all(r.passed for r in results if r.stage != "ok")
        status = "[bright_green]PASS[/bright_green]" if passed else "[red]FAIL[/red]"
        table.add_row(
            br["filename"],
            sym(results, "syntax"),
            sym(results, "import"),
            sym(results, "pytest"),
            str(attempts),
            status,
        )

    console.print(Panel(
        table,
        title=Text("  Verification Results  ", style=THEME["accent"]),
        title_align="left",
        border_style="dark_orange3",
    ))
    console.print()


def print_summary(summary: str, run_instructions: str, file_list: list[str]):
    """Render the final BUILD COMPLETE panel with architecture summary, file list, and run instructions."""
    console.print()
    console.print(Rule(f"[{THEME['accent']}]BUILD COMPLETE[/{THEME['accent']}]", style=THEME["rule"]))
    console.print()

    console.print(Panel(
        Text(wrap(summary), style="white"),
        title=Text("  Architecture  ", style=THEME["accent"]),
        title_align="left",
        border_style="dark_orange3",
        padding=(0, 2),
    ))
    console.print()

    # File list table
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style=THEME["dim"])
    table.add_column(style=THEME["filename"])
    for i, f in enumerate(file_list, 1):
        table.add_row(f"{i}.", f)
    console.print(Panel(table, title=Text("  Files generated  ", style=THEME["accent"]),
                        title_align="left", border_style="dark_orange3"))
    console.print()

    console.print(Panel(
        Text(run_instructions, style="grey74"),
        title=Text("  How to run  ", style=THEME["accent"]),
        title_align="left",
        border_style="dark_orange3",
        padding=(0, 2),
    ))
    console.print()

# ── Main ──────────────────────────────────────────────────────────────────────

def collect_brief_interactively() -> str:
    """
    Prompt the user to paste or type their brief when no --brief flag was given.
    Reads until the user enters a line containing only 'END' (case-insensitive)
    or sends EOF (Ctrl-D / Ctrl-Z).
    """
    console.print(Panel(
        Text.assemble(
            ("No brief supplied.\n\n", "bold white"),
            ("Paste or type your project brief below.\n", THEME["dim"]),
            ("Include: goal, hardware specs (GPU/VRAM/CPU/RAM/OS), constraints.\n", THEME["dim"]),
            ("When finished, enter ", THEME["dim"]),
            ("END", "bold cyan"),
            (" on a line by itself (or press Ctrl-D).", THEME["dim"]),
        ),
        border_style="grey30",
        padding=(0, 2),
    ))
    console.print()

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass  # Ctrl-D / pipe end

    brief = "\n".join(lines).strip()
    if not brief:
        console.print("[bold red]Error:[/bold red] Empty brief — nothing to build.")
        sys.exit(1)
    return brief


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous builder — turns any brief into a working codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python build.py --brief my_brief.txt
              python build.py --brief my_brief.txt --out ./output
              python build.py          # interactive brief entry
              python build.py --brief my_brief.txt --out ./output --resume
        """),
    )
    parser.add_argument("--brief", help="Path to a text file containing the brief")
    parser.add_argument("--out", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument(
        "--skip-feasibility", action="store_true",
        help="Skip Phase 0 feasibility check and go straight to the architect"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed even if the feasibility check returns 'infeasible'"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume a previous build from checkpoint. Restores sys_config, prior_specs, "
            "completed files, and iteration counter so the coder picks up where it left off "
            "without loss spikes or LR-schedule drift."
        )
    )
    args = parser.parse_args()

    # ── API key ──
    api_key = args.key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] Set ANTHROPIC_API_KEY or pass --key")
        sys.exit(1)

    # ── Brief ──
    if args.brief:
        brief_path = Path(args.brief)
        if not brief_path.exists():
            console.print(f"[bold red]Error:[/bold red] Brief file not found: {brief_path}")
            sys.exit(1)
        brief = brief_path.read_text(encoding="utf-8").strip()
    else:
        brief = collect_brief_interactively()

    # ── Output dir ──
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    # ── Improvement 6: initialise persistent file logging ──────────────────────
    log_path = setup_file_logging(out_dir)
    _build_logger.info(
        "BUILD START  model=%s  out=%s  resume=%s  skip_feasibility=%s",
        ARCHITECT_MODEL, out_dir, args.resume, args.skip_feasibility,
    )

    # ── Checkpoint restore (Change 7: stable resume behavior) ──────────────────
    # Restores: sys_config_json, prior_specs, completed_files, startup_commands,
    # model_stack_data, and iteration counter so the run continues from exactly
    # where it stopped — no schedule drift, no loss spikes on resume.
    _restored_checkpoint: dict | None = None
    if args.resume:
        _cp = load_checkpoint(out_dir)
        if _cp:
            _restored_checkpoint = _cp
            console.print(Panel(
                Text.assemble(
                    ("⟳  Resuming from checkpoint\n\n", "bold cyan"),
                    (f"  Timestamp : {_cp.get('timestamp', '—')}\n", THEME["dim"]),
                    (f"  Iteration : {_cp.get('iteration', '—')}\n", THEME["dim"]),
                    (f"  Completed : {len(_cp.get('completed_files', []))} file(s)\n", THEME["dim"]),
                    (f"  Pass rate : {_cp.get('best_pass_rate', '—')}\n", THEME["dim"]),
                ),
                border_style="cyan",
                padding=(0, 2),
            ))
            console.print()
        else:
            console.print(Padding(
                Text("⚠  --resume given but no valid checkpoint found. Starting fresh.", style="yellow"),
                (0, 4),
            ))
            console.print()

    client = anthropic.Anthropic(api_key=api_key)

    print_header()

    # ── Show brief ──
    console.print(Panel(
        Text(brief, style=THEME["dim"]),
        title=Text("  Brief  ", style="bold white"),
        title_align="left",
        border_style="grey30",
        padding=(0, 2),
    ))
    console.print()

    # ── Phase 0: Feasibility Check ──
    if not args.skip_feasibility:
        print_phase("PHASE 0  ·  feasibility check — can the hardware meet the brief?", "bold white")
        _build_logger.info("PHASE 0  feasibility check started")

        feasibility = run_feasibility_check(client, brief)
        print_feasibility_result(feasibility)

        if feasibility.status == "infeasible" and not args.force:
            console.print(Panel(
                Text.assemble(
                    ("The feasibility check determined this brief CANNOT be achieved\n", "bold red"),
                    ("on the described hardware.\n\n", "bold red"),
                    ("Blockers:\n", "bold white"),
                    *[(f"  • {b}\n", "red") for b in feasibility.blockers],
                    ("\nTo proceed anyway, re-run with ", THEME["dim"]),
                    ("--force", "bold cyan"),
                    (".\nTo adjust the brief, re-run with ", THEME["dim"]),
                    ("--brief <file>", "bold cyan"),
                    (".", THEME["dim"]),
                ),
                border_style="red",
                padding=(0, 2),
            ))
            sys.exit(1)

        if feasibility.status == "infeasible" and args.force:
            console.print(Padding(
                Text("⚠  --force supplied — proceeding despite infeasibility verdict.", style="yellow"),
                (0, 4)
            ))
            console.print()
    else:
        console.print(Padding(
            Text("⚡  Feasibility check skipped (--skip-feasibility).", style=THEME["dim"]),
            (0, 4)
        ))
        console.print()

    # ── Phase 1 + 2 + 3: Architect ──
    print_phase("ARCHITECT  ·  hardware audit → config → file specs", THEME["architect"])
    _build_logger.info("ARCHITECT phase started")

    pending_files: list[dict] = []
    finalize_data: dict | None = None
    file_specs_seen: set[str] = set()
    sys_config_json: str | None = None
    startup_commands: dict = {}   # stashed from hw_audit, used at launch time
    model_stack_data: dict = {}   # stashed for health-poll endpoint lookup
    # prior_specs is seeded with config.py's contract inside the hw_audit handler
    # so every coder call sees `from config import cfg` as an available import
    prior_specs: list[dict] = []
    # completed_files tracks filenames already written+verified (used by resume)
    completed_files: set[str] = set()

    # ── Restore state from checkpoint (Change 7) ────────────────────────────
    # Re-injects sys_config_json, prior_specs, startup_commands, model_stack,
    # and the set of already-completed files so the coder loop can skip them.
    # This prevents the step-counter / schedule mismatch that causes the
    # "loss drops then drifts back" symptom observed after interrupted restarts.
    if _restored_checkpoint:
        sys_config_json  = _restored_checkpoint.get("sys_config_json")
        startup_commands = _restored_checkpoint.get("startup_commands", {})
        model_stack_data = _restored_checkpoint.get("model_stack_data", {})
        prior_specs      = _restored_checkpoint.get("prior_specs", [])
        completed_files  = set(_restored_checkpoint.get("completed_files", []))
        console.print(Padding(
            Text(
                f"⟳  State restored — skipping architect phase "
                f"({len(completed_files)} file(s) already done).",
                style="cyan"
            ),
            (0, 4),
        ))
        console.print()
    else:
        for event in run_architect(client, brief):
            if event["type"] == "reasoning":
                print_reasoning(event["text"])

            elif event["type"] == "hw_audit":
                sys_config_json  = event["config_json"]
                startup_commands = event.get("startup_commands", {})
                model_stack_data = event.get("model_stack", {})

                # Change 4: pre-flight port conflict check — warn before any
                # container tries to bind and fails silently at runtime.
                port_warnings = check_ports_in_model_stack(model_stack_data)
                for pw in port_warnings:
                    console.print(Padding(Text(f"⚠  {pw}", style="yellow"), (0, 4)))

                # ── Write sys_config.json ──
                cfg_path = write_file(out_dir, "sys_config.json", sys_config_json)
                print_hw_audit(event["vram_allocation"], event["model_stack"],
                               event.get("startup_commands", {}), sys_config_json)
                console.print(f"  [{THEME['success']}]✓[/{THEME['success']}]  sys_config.json → {cfg_path}")

                # ── Write config.py deterministically ──
                # Generated by us, not the coder model — guaranteed correct.
                config_code = generate_config_py(sys_config_json)
                config_path = write_file(out_dir, "config.py", config_code)

                # Verify it parses and imports cleanly right now
                syntax_ok = check_syntax(config_code, "config.py")
                import_ok = check_imports(config_path, out_dir) if syntax_ok.passed else syntax_ok
                if not syntax_ok.passed or not import_ok.passed:
                    err = (syntax_ok if not syntax_ok.passed else import_ok).error
                    console.print(f"  [bold red]✗  config.py failed verification:[/bold red]\n{err}")
                    sys.exit(1)

                console.print(f"  [{THEME['success']}]✓[/{THEME['success']}]  config.py → {config_path}  [grey42](syntax ✓  import ✓)[/grey42]")
                console.print()

                # Seed prior_specs so every coder call sees `from config import cfg`
                # as an available import before it writes a single line
                prior_specs.append({
                    "filename": "config.py",
                    "api_contract": CONFIG_PY_CONTRACT,
                })

            elif event["type"] == "file_spec":
                fn = event["filename"]
                if fn not in file_specs_seen:
                    file_specs_seen.add(fn)
                    pending_files.append({"filename": fn, "spec": event["spec"]})
                    print_file_spec(fn, event["spec"])

            elif event["type"] == "phase_error":
                print_phase_error(event["message"])

            elif event["type"] == "finalize":
                finalize_data = event

    # ── Guard: architect must have produced a config and at least one file ──
    # (Change 9: safety guards — fail fast with a clear message rather than
    #  silently producing a broken build)
    if sys_config_json is None:
        console.print("[bold red]Architect never produced a hardware config. Exiting.[/bold red]")
        sys.exit(1)

    if not pending_files and not completed_files:
        console.print("[bold red]Architect produced no files. Exiting.[/bold red]")
        sys.exit(1)

    # ── Coder + Verify ──
    console.print()
    print_phase("CODER  ·  generate → verify → fix", THEME["coder"])
    _build_logger.info("CODER phase started  total_files=%d", len(pending_files))

    written: list[tuple[str, Path]] = []
    build_results: list[dict] = []   # for the final verification table
    total_files   = len(pending_files)
    passed_count  = 0

    for item in pending_files:
        fn, spec = item["filename"], item["spec"]

        # ── Resume skip (Change 7): skip files already completed in a prior run ──
        if fn in completed_files:
            console.print(
                f"  [{THEME['dim']}]↷  {fn}  skipped (already completed in prior run)[/{THEME['dim']}]"
            )
            written.append((fn, out_dir / fn))
            build_results.append({
                "filename": fn,
                "results": [VerifyResult(passed=True, stage="ok")],
                "attempts": 0,
            })
            passed_count += 1
            continue

        print_generating(fn)

        # Generate initial version
        code = generate_file(client, fn, spec, prior_specs, sys_config_json, out_dir)

        # ── Change 6: empty-code recovery at generation stage ───────────────
        # If the initial generation came back empty, attempt a fix with an
        # explicit "you returned empty output" prompt before giving up.
        if _code_is_empty(code):
            console.print(
                f"  [yellow]⚠  {fn}  initial generation returned empty output "
                f"— attempting recovery[/yellow]"
            )
            empty_failure = VerifyResult(
                passed=False, stage="syntax",
                error=(
                    "The previous attempt resulted in an empty file. "
                    "Please output the FULL implementation code without truncation."
                ),
            )
            code = fix_file(
                client, fn, spec,
                "# (empty — initial generation produced no output)",
                empty_failure, prior_specs, sys_config_json, out_dir,
            )

        # Hard abort if still empty after the recovery attempt
        if _code_is_empty(code):
            console.print(
                f"  [red]✗  {fn}  generation still empty after recovery attempt — skipping file[/red]"
            )
            build_results.append({
                "filename": fn,
                "results": [VerifyResult(passed=False, stage="syntax",
                                         error="code generation returned empty output")],
                "attempts": 2,
            })
            continue

        filepath = write_file(out_dir, fn, code)

        # ── Improvement 5: auto-install deps when requirements.txt is generated ─
        if fn == "requirements.txt":
            install_requirements(out_dir)

        # ── Improvement 4: Incremental structural verification ───────────────
        # Validate the API_CONTRACT section of the spec against the generated
        # code before running the full verify_and_fix pipeline.  This catches
        # missing exports (classes, functions) early — before they can cause
        # cascading ImportErrors in files generated later in the sequence.
        _warn_if_contract_mismatch(fn, spec, code)

        # Verify and fix — up to MAX_RETRIES attempts
        final_code, results = verify_and_fix(
            client, fn, spec, code, filepath, out_dir,
            prior_specs, sys_config_json,
        )

        # Count how many generation/fix attempts happened
        # Each syntax failure that got fixed = one extra attempt
        attempts = 1 + sum(1 for r in results if not r.passed)

        print_file_written(fn, filepath, len(final_code.splitlines()))
        print_verify_result(results, fn, attempts)

        file_passed = all(r.passed for r in results if r.stage != "ok")
        if file_passed:
            passed_count += 1

        build_results.append({"filename": fn, "results": results, "attempts": attempts})

        # Update prior_specs only with the successfully verified contract
        prior_specs.append({
            "filename": fn,
            "api_contract": _extract_api_contract(spec),
        })
        written.append((fn, filepath))
        completed_files.add(fn)

        # Show a syntax-highlighted preview (first 25 lines)
        preview = "\n".join(final_code.splitlines()[:25])
        if len(final_code.splitlines()) > 25:
            preview += f"\n# … {len(final_code.splitlines()) - 25} more lines"
        console.print(Padding(
            Syntax(preview, lang_for(fn), theme="monokai", line_numbers=False,
                   background_color="default"),
            (0, 4, 1, 4)
        ))

        # ── Conditional checkpoint save (Changes 1 + 9) ──────────────────────
        # Only save when the file passed all checks — mirrors "save best checkpoint"
        # behaviour. This prevents a corrupt mid-build state from masquerading as
        # a valid resume point after a crash.
        best_pass_rate = passed_count / total_files if total_files else 0.0
        if file_passed:
            save_checkpoint(out_dir, {
                "sys_config_json":  sys_config_json,
                "startup_commands": startup_commands,
                "model_stack_data": model_stack_data,
                "prior_specs":      prior_specs,
                "completed_files":  list(completed_files),
                "best_pass_rate":   round(best_pass_rate, 4),
                "iteration":        len(completed_files),
            })

    # ── Verification summary table ──
    console.print()
    print_phase("VERIFICATION SUMMARY", THEME["accent"])
    print_verification_summary(build_results)

    failed = [br for br in build_results if not all(r.passed for r in br["results"] if r.stage != "ok")]
    if failed:
        console.print(Padding(
            Text(
                f"⚠  {len(failed)} file(s) could not be fully verified after {MAX_RETRIES} attempts.\n"
                f"   Check build.log for details. The files are written but may need manual fixes.",
                style="yellow"
            ),
            (0, 4)
        ))
        console.print()

    # ── Summary ──
    if finalize_data:
        print_summary(
            finalize_data.get("summary", ""),
            finalize_data.get("run_instructions", ""),
            finalize_data.get("file_list", [f for f, _ in written]),
        )
    else:
        console.print()
        console.print(Rule(f"[{THEME['success']}]DONE[/{THEME['success']}]", style=THEME["rule"]))
        console.print()
        for fn, fp in written:
            console.print(f"  [{THEME['success']}]✓[/{THEME['success']}]  {fp}")
        console.print()

    # ── Improvement 6: write structured build summary to the persistent log ─────
    # The FileHandler attached by setup_file_logging() has already captured every
    # LLM call, verification stage, checkpoint save, and phase transition during
    # the run.  Here we append a concise summary so the log file is useful as a
    # standalone audit record even without scrolling through the full debug output.
    _build_logger.info("BUILD COMPLETE  files_written=%d", len(written))
    for br in build_results:
        stages  = {r.stage: ("PASS" if r.passed else "FAIL") for r in br["results"]}
        overall = "PASS" if all(r.passed for r in br["results"] if r.stage != "ok") else "FAIL"
        _build_logger.info(
            "RESULT  %-40s  %s  syntax=%-4s  import=%-4s  pytest=%-4s  attempts=%d",
            br["filename"], overall,
            stages.get("syntax", "—"),
            stages.get("import", "—"),
            stages.get("pytest", "—"),
            br["attempts"],
        )
        for r in br["results"]:
            if not r.passed:
                _build_logger.warning(
                    "ERROR  %s  stage=%s\n%s",
                    br["filename"], r.stage,
                    "\n".join(r.error.splitlines()[:15]),
                )
    console.print(f"[{THEME['dim']}]Build log: {log_path}[/{THEME['dim']}]")
    console.print()

    # ── Launch models ──
    # Run after everything else is written and logged so a launch failure
    # doesn't prevent the build artefacts from being saved.
    if startup_commands:
        launch_all_models(model_stack_data, startup_commands)
    else:
        console.print(
            f"[{THEME['dim']}]No startup commands found in config — "
            f"start models manually.[/{THEME['dim']}]"
        )
        console.print()


if __name__ == "__main__":
    main()
