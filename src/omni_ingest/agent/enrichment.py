"""Agents for augmenting knowledge items with metadata, summaries, and thematic tags."""

import json
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..core.config import settings
from ..core.model import IngestionContext, Step, StepResult, StepStatus
from ..core.pipeline import register_step


def _get_openai_client(provider: str):
    if provider == "azure":
        key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else None
        return AsyncAzureOpenAI(
            api_key=key,
            api_version="2024-02-01",
            azure_endpoint=str(settings.azure_openai_endpoint) if settings.azure_openai_endpoint else "",
        )
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return AsyncOpenAI(api_key=key)


@register_step("enrichment")
class DomainEnrichmentAgent(BaseModel, Step):
    """
    Augments KnowledgeItems with generated summaries and thematic tags.

    This agent utilizes an LLM to analyze the content of each item and
    produce concise summaries and relevant keywords, facilitating
    improved categorization and searchability.
    """

    model_config = {"arbitrary_types_allowed": True}
    name: str = Field("enrichment", description="Internal name for this enrichment step")
    provider: str = Field("openai", description="The AI service to use (e.g., 'openai' or 'azure')")
    model: str = Field("gpt-4o", description="The AI model version used for analysis")
    prompt_template: str = Field("Analyze text and provide JSON summary/tags. Content: {text}", description="LLM prompt template")
    response_format: str = Field("json_object", description="The technical format the AI should use for its answer")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items to enrich")
        client = self._get_client()
        for item in ctx.items:
            prompt = self.prompt_template.format(text=item.content)
            try:
                res = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": self.response_format},
                )
                if self.response_format == "json_object":
                    data = json.loads(res.choices[0].message.content)
                    item.metadata.update(data)
                else:
                    item.metadata["enrichment"] = res.choices[0].message.content
            except Exception as e:
                return StepResult(status=StepStatus.FAILURE, error=f"Enrichment failed: {e}")
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"enriched_count": len(ctx.items)})


@register_step("learning_outcome")
class LearningOutcomeExtractionAgent(BaseModel, Step):
    """
    Extracts educational learning outcomes from instructional content.

    This agent identifies and extracts specific learning objectives or
    outcomes from text designated as educational, providing a structured
    representation of teaching goals.
    """

    model_config = {"arbitrary_types_allowed": True}
    name: str = Field("learning_outcome", description="Internal step name")
    provider: str = Field("openai", description="The AI service to use")
    model: str = Field("gpt-4o", description="The AI model used for extraction")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        if ctx.domain_profile != "education":
            return StepResult(status=StepStatus.SKIPPED, error="Not an education domain")
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for LO extraction")

        client = self._get_client()
        prompt_tmpl = (
            "Extract a list of learning outcomes from the following educational content. "
            "Return a JSON array of strings.\n\nContent: {text}"
        )

        for item in ctx.items:
            try:
                res = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_tmpl.format(text=item.content)}],
                    response_format={"type": "json_object"},
                )
                data = json.loads(res.choices[0].message.content)
                lo = data.get("learning_outcomes") or data.get("outcomes") or list(data.values())[0]
                if isinstance(lo, list):
                    item.metadata["learning_outcomes"] = lo
            except Exception as e:
                return StepResult(status=StepStatus.FAILURE, error=f"LO extraction failed: {e}")
        return StepResult(status=StepStatus.SUCCESS, items=ctx.items, metadata={"extracted_for": len(ctx.items)})


@register_step("kg_builder")
class KnowledgeGraphBuilderAgent(BaseModel, Step):
    """
    Identifies and extracts relational data for knowledge graph construction.

    This agent analyzes text to identify entities and their interrelationships,
    structuring the output into nodes and edges suitable for injection
    into a graph database.
    """

    model_config = {"arbitrary_types_allowed": True}
    name: str = Field("kg_builder", description="Internal step name")
    provider: str = Field("openai", description="The AI service to use")
    model: str = Field("gpt-4o", description="The AI model used for graph building")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for KG extraction")

        client = self._get_client()
        prompt_tmpl = (
            "Directly extract entities and their relationships from the following text into "
            "a graph format. Return JSON with 'nodes' (id, label, properties) and 'edges' "
            "(source, target, type).\n\nContent: {text}"
        )

        all_nodes, all_edges = [], []
        for item in ctx.items:
            try:
                res = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_tmpl.format(text=item.content)}],
                    response_format={"type": "json_object"},
                )
                data = json.loads(res.choices[0].message.content)
                all_nodes.extend(data.get("nodes", []))
                all_edges.extend(data.get("edges", []))
            except Exception as e:
                return StepResult(status=StepStatus.FAILURE, error=f"KG extraction failed: {e}")

        ctx.metadata["kg_data"] = {"nodes": all_nodes, "edges": all_edges}
        return StepResult(status=StepStatus.SUCCESS, metadata={"nodes": len(all_nodes), "edges": len(all_edges)})


@register_step("faq_generation")
class FAQGenerationAgent(BaseModel, Step):
    """
    Generates frequently asked questions (FAQs) from extracted content.

    This agent utilizes an LLM to analyze the primary text of KnowledgeItems
    and produce a structured set of questions and answers, facilitating
    automated FAQ generation for finalized knowledge bases.
    """

    model_config = {"arbitrary_types_allowed": True}
    name: str = Field("faq_generation", description="Internal name for the FAQ generation step")
    provider: str = Field("openai", description="The AI service used for generation")
    model: str = Field("gpt-4o", description="The AI model version used for generation")

    _client: Any = PrivateAttr(default=None)

    def _get_client(self):
        if self._client is None:
            self._client = _get_openai_client(self.provider)
        return self._client

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not ctx.items:
            return StepResult(status=StepStatus.SKIPPED, error="No items for FAQ generation")
        combined = "\n".join(i.content[:1000] for i in ctx.items[:5])
        prompt = (
            "Generate 3-5 frequently asked questions and answers based on the following text. "
            "Return a JSON object containing an 'faqs' key with an array of objects "
            "mapping 'q' to the question and 'a' to the answer.\n\n"
            f"Content: {combined}"
        )
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates FAQs in JSON format."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            faqs = data.get("faqs", [])
            ctx.metadata["faqs"] = faqs
            return StepResult(status=StepStatus.SUCCESS, metadata={"faq_count": len(faqs)})
        except Exception as e:
            return StepResult(status=StepStatus.FAILURE, error=f"FAQ generation failed: {e}")
