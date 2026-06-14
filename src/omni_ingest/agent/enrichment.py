"""Agents with behavior beyond item-level structured metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.pipeline import IngestionContext

from ..core.model import AgentMixin, Step, StepResult, StepStatus


class KGNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    source: str
    target: str
    type: str


class KGData(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


class KnowledgeGraphBuilderAgent(BaseModel, Step, AgentMixin):
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for KG extraction")

        agent = self._agent(ctx)
        all_nodes, all_edges = [], []
        for item in ctx.items:
            res = await agent.run("Directly extract entities and their relationships from the following text into a graph format. Content: " + item.decode(), output_type=KGData)
            all_nodes.extend([n.model_dump() for n in res.output.nodes])
            all_edges.extend([e.model_dump() for e in res.output.edges])

        ctx.metadata["kg_data"] = {"nodes": all_nodes, "edges": all_edges}
        return StepResult(status=StepStatus.SUCCESS, metadata={"nodes": len(all_nodes), "edges": len(all_edges)})


class FAQItem(BaseModel):
    q: str = Field(..., description="The question")
    a: str = Field(..., description="The answer")


class FAQGenerationAgent(BaseModel, Step, AgentMixin):
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for FAQ generation")

        text = "\n".join(i.decode()[:1000] for i in ctx.items[:5])
        res = await self._agent(ctx).run(
            f"Generate 3-5 frequently asked questions and answers based on the following text.\n\nContent: {text}",
            output_type=list[FAQItem],
            instructions="You are a helpful assistant that generates FAQs in JSON format.",
        )
        ctx.metadata["faqs"] = [f.model_dump() for f in res.output]
        return StepResult(status=StepStatus.SUCCESS, metadata={"faq_count": len(ctx.metadata["faqs"]), "faqs": ctx.metadata["faqs"]})
