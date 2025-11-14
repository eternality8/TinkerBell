"""Smoke tests for the AI controller scaffolding."""

import asyncio

from tinkerbell.ai.agents.executor import AIController


def test_ai_controller_returns_payload(sample_snapshot):
    controller = AIController(graph={"nodes": []})

    async def run() -> dict:
        return await controller.run_chat("hi", sample_snapshot)

    result = asyncio.run(run())
    assert result["prompt"] == "hi"
