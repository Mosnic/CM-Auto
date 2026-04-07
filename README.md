CM-Auto: Autonomous Builder
CM-Auto is a production-grade, multi-phase autonomous agent designed to turn a project brief into a fully functional, verified codebase. It handles hardware feasibility analysis, system architecture, model selection for ROCm/CUDA/Ollama, code generation, and automated self-healing through a "Syntax → Import → Pytest" verification pipeline.

Key Features
Phase 0 Feasibility Audit: Analyzes your project brief against real-time hardware telemetry (VRAM, CPU, RAM) to ensure the goal is achievable before a single line of code is written.

Intelligent Model Orchestration: Automatically selects and launches optimized models (e.g., Qwen3-VL for vision, DeepSeek for coding) based on available VRAM and performance needs.

Self-Healing Coder: If the generated code fails syntax checks, import tests, or unit tests, a specialized "Fixer" agent inspects the error and applies corrections automatically.

Robust Lifecycle Management: Features atomic checkpointing for resuming interrupted builds and automatic Docker container cleanup to prevent GPU "zombie" processes.

Local-First AI: Built specifically for local inference environments, supporting vLLM (ROCm/CUDA) and Ollama with sequential loading to prevent VRAM contention.

System Requirements
OS: Ubuntu 24.04 LTS (recommended).

Hardware: AMD (ROCm) or NVIDIA (CUDA) GPU.

Python: 3.11+.

Dependencies: anthropic, rich, tiktoken, and claude-agent-sdk.

📖 Usage
1. Installation
Bash
pip install anthropic rich tiktoken claude-agent-sdk
export ANTHROPIC_API_KEY=sk-ant-...
2. Start a Build
You can provide a brief via a text file or enter it interactively:

Bash
# Using a file
python build.py --brief example_brief.txt --out ./my_project

# Interactive mode
python build.py --out ./my_project
3. Resume an Interrupted Build
If a build is interrupted, use the --resume flag to pick up exactly where the agent left off without losing progress:

Bash
python build.py --out ./my_project --resume
🏗️ Architecture
The system operates in four distinct stages:

Analyst (Phase 0): Checks if the hardware (e.g., 32GB VRAM) can handle the requested AI workloads (Vision, Embeddings, etc.).

Architect (Phase 1): Locks down the sys_config.json, assigns ports, and specifies the file structures.

Coder (Phase 2): Implements the files using config.py as a single source of truth for all paths and endpoints.

Verifier (Phase 3): Runs AST parsing, subprocess import checks, and pytest. It triggers the Fixer loop upon any failure.

📂 Project Structure Example
When CM-Auto finishes a build (such as the cat monitor example), it generates:

sys_config.json: The hardware/model map.

config.py: The deterministic config loader used by all modules.

build.log: A persistent audit trail of all LLM reasoning and test results.

tests/: Automatically generated pytest suites for every component.

⚠️ Security
CM-Auto includes a shell-injection guard that refuses to execute model startup commands containing dangerous operators (;, &&, ||, etc.) to ensure local system integrity.
