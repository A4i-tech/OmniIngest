"""Agents for augmenting knowledge items with metadata, summaries, and thematic tags."""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from ..core.model import IngestionContext, Step, StepResult, StepStatus, handle_step_error
from ..core.pipeline import step


class DomainEnrichmentResult(BaseModel):
    summary: str
    tags: list[str]


@step("enrichment")
class DomainEnrichmentAgent(BaseModel, Step):
    """
    Augments KnowledgeItems with generated summaries and thematic tags.

    This agent utilizes an LLM to analyze the content of each item and
    produce concise summaries and relevant keywords, facilitating
    improved categorization and searchability.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for analysis")
    prompt_template: str = Field("Analyze text and provide JSON summary/tags. Content: {text}", description="LLM prompt template")
    response_format: str = Field("json_object", description="The technical format the AI should use for its answer")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to enrich")
        agent = Agent(self.model, output_type=DomainEnrichmentResult)
        for item in ctx.items:
            prompt = self.prompt_template.format(text=item.content)
            res = await agent.run(prompt)
            item.metadata.update(res.output.model_dump())
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"enriched_count": len(ctx.items)})


class LearningOutcomes(BaseModel):
    learning_outcomes: list[str] = Field(..., description="A list of learning outcomes")


@step("learning_outcome")
class LearningOutcomeExtractionAgent(BaseModel, Step):
    """
    Extracts educational learning outcomes from instructional content.

    This agent identifies and extracts specific learning objectives or
    outcomes from text designated as educational, providing a structured
    representation of teaching goals.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for extraction")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if ctx.domain_profile != "education":
            return StepResult(status=StepStatus.SKIPPED, error="Not an education domain")
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for LO extraction")

        agent = Agent(self.model, output_type=LearningOutcomes)
        prompt_tmpl = "Extract a list of learning outcomes from the following educational content. Content: {text}"

        for item in ctx.items:
            res = await agent.run(prompt_tmpl.format(text=item.content))
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


@step("kg_builder")
class KnowledgeGraphBuilderAgent(BaseModel, Step):
    """
    Identifies and extracts relational data for knowledge graph construction.

    This agent analyzes text to identify entities and their interrelationships,
    structuring the output into nodes and edges suitable for injection
    into a graph database.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for graph building")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for KG extraction")

        agent = Agent(self.model, output_type=KGData)
        prompt_tmpl = "Directly extract entities and their relationships from the following text into a graph format. Content: {text}"

        all_nodes, all_edges = [], []
        for item in ctx.items:
            res = await agent.run(prompt_tmpl.format(text=item.content))
            data = res.output
            all_nodes.extend([n.model_dump() for n in data.nodes])
            all_edges.extend([e.model_dump() for e in data.edges])

        ctx.metadata["kg_data"] = {"nodes": all_nodes, "edges": all_edges}
        return StepResult(status=StepStatus.SUCCESS, metadata={"nodes": len(all_nodes), "edges": len(all_edges)})


class FAQItem(BaseModel):
    q: str = Field(..., description="The question")
    a: str = Field(..., description="The answer")


class FAQList(BaseModel):
    faqs: list[FAQItem]


@step("faq_generation")
class FAQGenerationAgent(BaseModel, Step):
    """
    Generates frequently asked questions (FAQs) from extracted content.

    This agent utilizes an LLM to analyze the primary text of KnowledgeItems
    and produce a structured set of questions and answers, facilitating
    automated FAQ generation for finalized knowledge bases.
    """

    model: KnownModelName = Field("openai:gpt-4o", description="Model to use for generation")

    @handle_step_error
    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for FAQ generation")

        agent = Agent(self.model, output_type=FAQList, system_prompt="You are a helpful assistant that generates FAQs in JSON format.")
        combined = "\n".join(i.content[:1000] for i in ctx.items[:5])
        prompt = f"Generate 3-5 frequently asked questions and answers based on the following text.\n\nContent: {combined}"
        res = await agent.run(prompt)
        data = res.output
        faqs = [f.model_dump() for f in data.faqs]
        ctx.metadata["faqs"] = faqs
        return StepResult(status=StepStatus.SUCCESS, metadata={"faq_count": len(faqs)})
