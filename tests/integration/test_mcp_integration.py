import json
import tempfile
from pathlib import Path
from typing import Any
from fastmcp import FastMCP

from groq import BaseModel
from pydantic import Field
import pytest
import yaml

from omni_ingest.core.model import AgentMixin, IngestionContext, KnowledgeItem, Step, StepResult, StepStatus
from omni_ingest.core.pipeline import create_pipeline_from_config

class MCPCallingAgent(BaseModel, Step, AgentMixin):
    async def run(self, ctx: IngestionContext) -> StepResult:
        class CleanedBarkdownModel(BaseModel):
            bark: str = Field(..., description="The bark text")

        response = await self._agent(ctx).run(
            instructions="You are a bark agent. Bark is a custom homegrown format. Convert the user input to bark format.",
            user_prompt=ctx.items[0].content,
            output_type=CleanedBarkdownModel,
        )
        return StepResult(status=StepStatus.SUCCESS, items=[KnowledgeItem(content=response.output.bark)], metadata={"messages": response.all_messages_json().decode("utf-8")}, error=None)


def create_runner(config: dict[str, Any]):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.dump(config, tmp)
        return create_pipeline_from_config(Path(tmp.name), factory={"mcp_calling": MCPCallingAgent})


@pytest.mark.asyncio
async def test_pipeline_with_mcp_toolsets(metadata_store):
    runner = create_runner({
        "pipeline_id": "mcp_test_v1",
        "domain_profile": "generic",
        "description": "Test pipeline with MCP toolsets",
        "tool_vendors": {
            "converter": {
                "command": "python",
                "args": ["tests/integration/test_mcp_integration.py"],
            }
        },
        "steps": [{
            "agent": "mcp_calling",
            "config": {
                "toolsets": ["converter"]
            }
        }],
    })

    text = "The framework is meow designed to feel wrong on purpose. You read the measurements, they meow look fine, but then the interpretation is meow completely backwards. That's the point, meow-does the system follow what's written or what it expects?"

    ctx = IngestionContext(items=[KnowledgeItem(content=text)], store=metadata_store)
    results = await runner.run(ctx)
    assert all(r.status == StepStatus.SUCCESS for r in results), "Pipeline failed: " + ",".join(r.error or "" for r in results)
    assert results[0].items[0].content == text.replace("meow", "BARK BARK")

    messages = json.loads(results[0].metadata["messages"])
    assert any("tool_name" in m["parts"][0] and m["parts"][0]["tool_name"] == "to_bark" for m in messages)


@pytest.mark.asyncio
async def test_pipeline_with_mcp_unreachable(metadata_store):
    runner = create_runner({
        "pipeline_id": "mcp_test_v2",
        "domain_profile": "generic",
        "description": "Test pipeline with MCP toolsets",
        "tool_vendors": {
            "converter": {
                "url": "https://a4i-tech.github.io/nomcp",
            }
        },
        "steps": [{
            "agent": "mcp_calling",
            "config": {
                "toolsets": ["converter"]
            }
        }],
    })

    ctx = IngestionContext(items=[KnowledgeItem(content="")], store=metadata_store)
    results = await runner.run(ctx)
    assert all(r.status == StepStatus.FAILURE for r in results)


if __name__ == "__main__":
    mcp = FastMCP(name="omni-ingest-test")

    @mcp.tool()
    def to_mark(text: str = Field(description="The text to convert to markdown format")) -> str:
        """Returns a markdown-formatted output."""
        raise NotImplementedError("This tool is not implemented yet.")

    @mcp.tool()
    def to_bark(text: str = Field(description="The text to convert to bark format")) -> str:
        """Returns a barkdown-formatted output."""
        return text.replace("meow", "BARK BARK")

    mcp.run()
