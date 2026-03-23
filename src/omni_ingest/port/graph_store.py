from typing import Any

from ..core.protocol import GraphStore


class InMemoryGraphStore(GraphStore):
    """
    Volatile in-memory implementation of a graph store for testing and local dev.
    """

    def __init__(self):
        self.nodes = {}
        self.edges = []

    async def upsert_nodes(self, nodes: list[dict[str, Any]]):
        for n in nodes:
            if "id" in n:
                self.nodes[n["id"]] = n

    async def upsert_edges(self, edges: list[dict[str, Any]]):
        self.edges.extend(edges)

    async def close(self) -> None:
        pass


class Neo4jGraphStore(GraphStore):
    """
    Production graph storage using Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import AsyncGraphDatabase
        except ModuleNotFoundError as exc:
            raise ImportError("Neo4jGraphStore requires neo4j. Install omni-ingest[storage] or add neo4j to the environment.") from exc

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def upsert_nodes(self, nodes: list[dict[str, Any]]):
        rows = []
        for node in nodes:
            if "id" not in node:
                continue

            properties: dict[str, Any] = {}
            if "properties" in node and isinstance(node["properties"], dict):
                properties.update(node["properties"])
            for key, value in node.items():
                if key not in {"id", "label", "properties"}:
                    properties[key] = value

            label = "Entity"
            if "label" in node and isinstance(node["label"], str) and node["label"] != "":
                label = node["label"]

            rows.append({"id": str(node["id"]), "label": label, "properties": properties})

        if not rows:
            return

        query = """
        UNWIND $nodes AS node
        MERGE (n:Entity {id: node.id})
        SET n += node.properties
        SET n.label = node.label
        """
        async with self._driver.session() as session:
            result = await session.run(query, nodes=rows)
            await result.consume()

    async def upsert_edges(self, edges: list[dict[str, Any]]):
        rows = []
        for edge in edges:
            if "source" not in edge or "target" not in edge or "type" not in edge:
                continue

            properties = {}
            for key, value in edge.items():
                if key not in {"source", "target", "type"}:
                    properties[key] = value

            rows.append(
                {
                    "source": str(edge["source"]),
                    "target": str(edge["target"]),
                    "type": str(edge["type"]),
                    "properties": properties,
                }
            )

        if not rows:
            return

        query = """
        UNWIND $edges AS edge
        MERGE (source:Entity {id: edge.source})
        MERGE (target:Entity {id: edge.target})
        MERGE (source)-[rel:RELATED_TO {source: edge.source, target: edge.target, type: edge.type}]->(target)
        SET rel += edge.properties
        """
        async with self._driver.session() as session:
            result = await session.run(query, edges=rows)
            await result.consume()

    async def close(self) -> None:
        await self._driver.close()
