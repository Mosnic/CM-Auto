import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from config import cfg
from tools.frame_extractor import extract_frames_from_video
from tools.vision import VisionTool
from tools.cat_identity import CatIdentityEngine

# ---------------------------------------------------------------------------
# Tool definitions advertised to the coding model
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "extract_best_frames",
        "description": "Extract the best frames from a video clip",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to video file"}
            },
            "required": ["video_path"],
        },
    },
    {
        "name": "analyze_cat_image",
        "description": "Analyze an image for cat presence and characteristics",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to image file"}
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "process_visit",
        "description": "Process a cat visit to determine identity and update profile",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to cat image"},
                "analysis": {"type": "object", "description": "Vision analysis results"},
            },
            "required": ["image_path", "analysis"],
        },
    },
]


def create_initial_prompt(video_path: str) -> str:
    """Create initial prompt for video processing.

    Args:
        video_path: Path to video file

    Returns:
        str: Formatted initial prompt
    """
    return (
        f"You are an autonomous cat monitoring agent. A new video clip has arrived: {video_path}\n\n"
        "Your task is to:\n"
        "1. Extract the best frames showing cats clearly\n"
        "2. Analyze each frame for cat presence and characteristics\n"
        "3. Process any cat visits to determine identity and update profiles\n\n"
        "You have access to these tools:\n"
        "- extract_best_frames(video_path) - Extract top frames from video\n"
        "- analyze_cat_image(image_path) - Get structured cat analysis from image\n"
        "- process_visit(image_path, analysis) - Identify cat and update profile\n\n"
        "Start by extracting frames from the video. Respond with your tool calls."
    )


class AgentLoop:
    """Tool-use reasoning agent for processing cat monitoring video clips."""

    def __init__(self) -> None:
        """Initialize agent with tools and model endpoint."""
        self.coding_endpoint = cfg["models"]["coding"]["endpoint"]
        self.vision_tool = VisionTool()
        self.identity_engine = CatIdentityEngine()
        self.logger = logging.getLogger(__name__)
        self.max_iterations: int = cfg["thresholds"]["max_agent_iterations"]

    # ------------------------------------------------------------------
    # Tool schema
    # ------------------------------------------------------------------

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for the agent.

        Returns:
            List[Dict[str, Any]]: Tool definitions for the coding model
        """
        return TOOL_DEFINITIONS

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    async def call_coding_model(self, messages: List[Dict[str, str]]) -> str:
        """Call coding model with conversation history.

        Args:
            messages: Conversation history in OpenAI format

        Returns:
            str: Model response text
        """
        url = f"{self.coding_endpoint}/api/chat"
        payload = {
            "model": cfg["models"]["coding"]["model_id"],
            "messages": messages,
            "stream": False,
            "tools": self.get_available_tools(),
        }

        def _post() -> requests.Response:
            return requests.post(url, json=payload, timeout=120)

        # Retry once on network/timeout failure
        for attempt in range(2):
            try:
                response = await asyncio.get_event_loop().run_in_executor(None, _post)
                response.raise_for_status()
                data = response.json()
                # Ollama /api/chat returns {"message": {"content": "..."}}
                content: str = data.get("message", {}).get("content", "")
                return content
            except requests.exceptions.Timeout:
                if attempt == 0:
                    self.logger.warning("Coding model timed out; retrying once ...")
                    await asyncio.sleep(2)
                    continue
                self.logger.error("Coding model timed out after retry")
                raise
            except requests.exceptions.RequestException as exc:
                if attempt == 0:
                    self.logger.warning("Coding model request failed (%s); retrying ...", exc)
                    await asyncio.sleep(2)
                    continue
                self.logger.error("Coding model request failed: %s", exc)
                raise

        # Should be unreachable, but satisfies type checkers
        raise RuntimeError("call_coding_model: exhausted retries")

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model response.

        The model may emit tool calls as a JSON block:
            {"tool": "...", "arguments": {...}}
        or wrapped in a markdown fence.  Both forms are handled.

        Args:
            response: Raw model response text

        Returns:
            Optional[Dict[str, Any]]: Parsed tool call or None if no tool use
        """
        if not response:
            return None

        # Strip optional markdown fences
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first (```json or ```) and last (```) lines
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            text = inner.strip()

        # Look for a JSON object anywhere in the text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        json_str = text[start : end + 1]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self.logger.debug("JSON parse error in model response: %s | text=%r", exc, json_str)
            return None

        # Accept {"tool": ..., "arguments": ...} or {"name": ..., "arguments": ...}
        tool_name = parsed.get("tool") or parsed.get("name")
        arguments = parsed.get("arguments") or parsed.get("parameters") or {}

        if not tool_name:
            return None

        known_names = {t["name"] for t in TOOL_DEFINITIONS}
        if tool_name not in known_names:
            self.logger.debug("Unknown tool name in model response: %s", tool_name)
            return None

        return {"tool": tool_name, "arguments": arguments}

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return results.

        Args:
            tool_call: Dict with keys "tool" and "arguments"

        Returns:
            Dict[str, Any]: Tool execution results (always has "success" key)
        """
        tool_name: str = tool_call.get("tool", "")
        arguments: Dict[str, Any] = tool_call.get("arguments", {})

        try:
            if tool_name == "extract_best_frames":
                return await self.extract_best_frames(arguments["video_path"])
            elif tool_name == "analyze_cat_image":
                return await self.analyze_cat_image(arguments["image_path"])
            elif tool_name == "process_visit":
                return await self.process_visit(
                    arguments["image_path"], arguments["analysis"]
                )
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except KeyError as exc:
            msg = f"Missing required argument for {tool_name}: {exc}"
            self.logger.error(msg)
            return {"success": False, "error": msg}
        except Exception as exc:  # noqa: BLE001
            msg = f"Tool {tool_name} raised an exception: {exc}"
            self.logger.exception(msg)
            return {"success": False, "error": msg}

    # ------------------------------------------------------------------
    # Individual tool implementations
    # ------------------------------------------------------------------

    async def extract_best_frames(self, video_path: str) -> Dict[str, Any]:
        """Tool: Extract best frames from video clip.

        Args:
            video_path: Path to video file

        Returns:
            Dict[str, Any]: Frame extraction results
        """
        try:
            frames = await extract_frames_from_video(video_path)
            if not frames:
                return {"success": False, "error": "No frames extracted", "frames": []}
            return {"success": True, "frames": frames, "count": len(frames)}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Frame extraction failed for %s: %s", video_path, exc)
            return {"success": False, "error": str(exc), "frames": []}

    async def analyze_cat_image(self, image_path: str) -> Dict[str, Any]:
        """Tool: Analyze cat image using vision model.

        Args:
            image_path: Path to image file

        Returns:
            Dict[str, Any]: Cat analysis results
        """
        try:
            analysis = await self.vision_tool.analyze_image(image_path)
            return {"success": True, "analysis": analysis}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Vision analysis failed for %s: %s", image_path, exc)
            return {"success": False, "error": str(exc)}

    async def process_visit(
        self, image_path: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tool: Process cat visit and determine identity.

        Args:
            image_path: Path to cat image
            analysis: Vision analysis results

        Returns:
            Dict[str, Any]: Visit processing results
        """
        try:
            result = await self.identity_engine.process_visit(image_path, analysis)
            return {"success": True, "visit": result}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Visit processing failed for %s: %s", image_path, exc)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Main reasoning loop
    # ------------------------------------------------------------------

    async def process_video_clip(self, video_path: Path) -> Dict[str, Any]:
        """Main method to process a complete video clip.

        Runs an agentic tool-use loop:
        1. Send initial prompt to coding model.
        2. Parse tool calls from the response.
        3. Execute tools and feed results back.
        4. Repeat until the model stops calling tools or max_iterations reached.

        Args:
            video_path: Path to video file

        Returns:
            Dict[str, Any]: Complete processing results
        """
        video_str = str(video_path)
        self.logger.info("AgentLoop starting for clip: %s", video_str)

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": create_initial_prompt(video_str)}
        ]

        tool_results: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            self.logger.debug("Agent iteration %d / %d", iterations, self.max_iterations)

            # ----------------------------------------------------------
            # Ask the model what to do next
            # ----------------------------------------------------------
            try:
                response_text = await self.call_coding_model(messages)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Coding model call failed: %s", exc)
                return {
                    "success": False,
                    "error": f"Model API failure: {exc}",
                    "video_path": video_str,
                    "tool_results": tool_results,
                    "iterations": iterations,
                }

            self.logger.debug("Model response (iter %d): %r", iterations, response_text[:300])

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": response_text})

            # ----------------------------------------------------------
            # Check for a tool call
            # ----------------------------------------------------------
            tool_call = self.parse_tool_call(response_text)

            if tool_call is None:
                # No tool call — model is done (or gave a plain-text answer)
                self.logger.info(
                    "Agent finished after %d iteration(s) (no tool call detected)", iterations
                )
                break

            # ----------------------------------------------------------
            # Execute the tool
            # ----------------------------------------------------------
            self.logger.info("Executing tool: %s", tool_call["tool"])
            result = await self.execute_tool(tool_call)
            tool_results.append({"tool": tool_call["tool"], "result": result})

            # Feed the tool result back as a user message so the model can continue
            result_text = json.dumps(result, default=str)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool '{tool_call['tool']}' returned:\n{result_text}\n\n"
                        "Continue with the next step, or stop if the task is complete."
                    ),
                }
            )

            # If the tool failed, let the model decide to retry or stop
            if not result.get("success", False):
                self.logger.warning(
                    "Tool %s reported failure: %s", tool_call["tool"], result.get("error")
                )
        else:
            self.logger.warning(
                "Agent reached max iterations (%d) without finishing", self.max_iterations
            )

        # Summarise outcomes
        cats_identified = [
            r["result"].get("visit", {})
            for r in tool_results
            if r["tool"] == "process_visit" and r["result"].get("success")
        ]

        return {
            "success": True,
            "video_path": video_str,
            "iterations": iterations,
            "tool_results": tool_results,
            "cats_identified": cats_identified,
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

async def process_video_clip(video_path: Path) -> Dict[str, Any]:
    """Convenience function to process video clip with agent.

    Args:
        video_path: Path to video file

    Returns:
        Dict[str, Any]: Processing results
    """
    agent = AgentLoop()
    return await agent.process_video_clip(video_path)
