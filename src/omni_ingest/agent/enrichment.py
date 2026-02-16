"""Agents for augmenting knowledge items with metadata, summaries, and thematic tags."""

from typing import Any

from pydantic import BaseModel, Field

from ..core.model import AgentMixin, IngestionContext, Step, StepResult, StepStatus


class DomainEnrichmentResult(BaseModel):
    summary: str
    tags: list[str]


class DomainEnrichmentAgent(BaseModel, Step, AgentMixin):
    """
    Augments KnowledgeItems with generated summaries and thematic tags.

    This agent utilizes an LLM to analyze the content of each item and
    produce concise summaries and relevant keywords, facilitating
    improved categorization and searchability.
    """

    prompt_template: str = Field("Analyze text and provide JSON summary/tags. Content: {text}", description="LLM prompt template")
    response_format: str = Field("json_object", description="Response format for LLM output")

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to enrich")
        agent = self._agent(ctx)
        for item in ctx.items:
            prompt = self.prompt_template.format(text=item.content)
            res = await agent.run(prompt, output_type=DomainEnrichmentResult)
            item.metadata.update(res.output.model_dump())
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"enriched_count": len(ctx.items)})


class LearningOutcomes(BaseModel):
    learning_outcomes: list[str] = Field(..., description="Extracted learning outcomes")


class LearningOutcomeExtractionAgent(BaseModel, Step, AgentMixin):
    """
    Extracts educational learning outcomes from instructional content.

    This agent identifies and extracts specific learning objectives or
    outcomes from text designated as educational, providing a structured
    representation of teaching goals.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if ctx.domain_profile != "education":
            return StepResult(status=StepStatus.SKIPPED, error="Not an education domain")
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for LO extraction")

        agent = self._agent(ctx)
        prompt_tmpl = "Extract a list of learning outcomes from the following educational content. Content: {text}"

        for item in ctx.items:
            res = await agent.run(prompt_tmpl.format(text=item.content), output_type=LearningOutcomes)
            item.metadata["learning_outcomes"] = res.output.learning_outcomes
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"extracted_for": len(ctx.items)})


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
    """
    Identifies and extracts relational data for knowledge graph construction.

    This agent analyzes text to identify entities and their interrelationships,
    structuring the output into nodes and edges suitable for injection
    into a graph database.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for KG extraction")

        agent = self._agent(ctx)
        prompt_tmpl = "Directly extract entities and their relationships from the following text into a graph format. Content: {text}"

        all_nodes, all_edges = [], []
        for item in ctx.items:
            res = await agent.run(prompt_tmpl.format(text=item.content), output_type=KGData)
            data = res.output
            all_nodes.extend([n.model_dump() for n in data.nodes])
            all_edges.extend([e.model_dump() for e in data.edges])

        ctx.metadata["kg_data"] = {"nodes": all_nodes, "edges": all_edges}
        return StepResult(status=StepStatus.SUCCESS, metadata={"nodes": len(all_nodes), "edges": len(all_edges)})


class FAQItem(BaseModel):
    q: str = Field(..., description="The question")
    a: str = Field(..., description="The answer")


class FAQGenerationAgent(BaseModel, Step, AgentMixin):
    """
    Generates frequently asked questions (FAQs) from extracted content.

    This agent utilizes an LLM to analyze the primary text of KnowledgeItems
    and produce a structured set of questions and answers, facilitating
    automated FAQ generation for finalized knowledge bases.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for FAQ generation")

        agent = self._agent(ctx)
        combined = "\n".join(i.content[:1000] for i in ctx.items[:5])
        prompt = f"Generate 3-5 frequently asked questions and answers based on the following text.\n\nContent: {combined}"
        res = await agent.run(prompt, output_type=list[FAQItem], instructions="You are a helpful assistant that generates FAQs in JSON format.")
        faqs = [f.model_dump() for f in res.output]
        ctx.metadata["faqs"] = faqs
        return StepResult(status=StepStatus.SUCCESS, metadata={"faq_count": len(faqs), "faqs": faqs})
