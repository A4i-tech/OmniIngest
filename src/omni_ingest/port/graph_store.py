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
        self.uri = uri
        self.auth = (user, password)
        # from neo4j import GraphDatabase
        # self.driver = GraphDatabase.driver(uri, auth=(user, password))

    async def upsert_nodes(self, nodes: list[dict[str, Any]]):
        # Implementation would use neo4j driver
        pass

    async def upsert_edges(self, edges: list[dict[str, Any]]):
        pass

    async def close(self) -> None:
        pass
