"""
Neo4j connection manager for GraphRAGv2.

Provides a single, reusable driver instance (connection pool) with:
- Context-manager support  (with neo4j_client() as client: ...)
- Health-check / ping
- Convenience wrappers for read / write queries
- Auto-close on process exit

Usage::

    from src.graph_loader.neo4j_client import Neo4jClient

    with Neo4jClient() as client:
        client.ping()                        # raises if unreachable
        rows = client.read("MATCH (n:Play) RETURN n.title AS title LIMIT 5")
        client.write("CREATE (n:Play {id: $id})", id="test")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Thin wrapper around the official neo4j driver.

    Thread-safe: the underlying driver maintains a connection pool,
    so a single Neo4jClient instance can be shared across threads/components.
    """

    def __init__(
        self,
        uri:      Optional[str] = None,
        user:     Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self._uri      = uri      or settings.neo4j.uri
        self._user     = user     or settings.neo4j.user
        self._password = password or settings.neo4j.password
        self._driver: Optional[Driver] = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the driver / connection pool (idempotent)."""
        if self._driver is not None:
            return
        logger.info("Connecting to Neo4j at %s (user=%s)", self._uri, self._user)
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        logger.info("Neo4j driver created")

    def close(self) -> None:
        """Close the driver and free all pooled connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def ping(self) -> None:
        """
        Verify connectivity.  Raises ``ConnectionError`` if unreachable.
        """
        self._ensure_connected()
        try:
            self._driver.verify_connectivity()  # type: ignore[union-attr]
            logger.info("Neo4j ping OK — %s", self._uri)
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Neo4j unreachable at {self._uri}: {exc}"
            ) from exc
        except AuthError as exc:
            raise ConnectionError(
                f"Neo4j authentication failed (user={self._user}): {exc}"
            ) from exc

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def read(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read-only Cypher query and return all rows as dicts.

        Args:
            cypher:   Cypher query string.
            params:   Optional parameter dict.
            database: Neo4j database name (None → default).

        Returns:
            List of row dicts (field-name → value).
        """
        self._ensure_connected()
        with self._driver.session(database=database) as session:  # type: ignore
            result = session.run(cypher, params or {})
            rows = [record.data() for record in result]
        logger.debug("READ returned %d rows | %s", len(rows), _truncate(cypher))
        return rows

    def write(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Execute a write Cypher query inside an auto-commit transaction.

        Args:
            cypher:   Cypher query string.
            params:   Optional parameter dict.
            database: Neo4j database name (None → default).
        """
        self._ensure_connected()
        with self._driver.session(database=database) as session:  # type: ignore
            session.run(cypher, params or {})
        logger.debug("WRITE OK | %s", _truncate(cypher))

    def write_tx(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Execute a write query inside an explicit managed transaction
        (auto-retry on transient failures).
        """
        self._ensure_connected()

        def _work(tx: Any) -> None:
            tx.run(cypher, params or {})

        with self._driver.session(database=database) as session:  # type: ignore
            session.execute_write(_work)
        logger.debug("WRITE_TX OK | %s", _truncate(cypher))

    @contextmanager
    def session(self, **kwargs: Any) -> Generator[Session, None, None]:
        """Yield a raw neo4j Session for advanced usage."""
        self._ensure_connected()
        with self._driver.session(**kwargs) as s:  # type: ignore
            yield s

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def count_nodes(self, label: Optional[str] = None) -> int:
        """Return number of nodes, optionally filtered by label."""
        cypher = (
            f"MATCH (n:{label}) RETURN count(n) AS cnt"
            if label
            else "MATCH (n) RETURN count(n) AS cnt"
        )
        rows = self.read(cypher)
        return int(rows[0]["cnt"]) if rows else 0

    def count_relationships(self) -> int:
        """Return total number of relationships in the graph."""
        rows = self.read("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return int(rows[0]["cnt"]) if rows else 0

    def get_schema_summary(self) -> Dict[str, Any]:
        """
        Return a lightweight schema summary:
        node labels + relationship types + counts.
        """
        node_labels  = self.read("CALL db.labels() YIELD label RETURN label")
        rel_types    = self.read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        total_nodes  = self.count_nodes()
        total_rels   = self.count_relationships()

        return {
            "node_labels":      [r["label"] for r in node_labels],
            "relationship_types": [r["relationshipType"] for r in rel_types],
            "total_nodes":      total_nodes,
            "total_relationships": total_rels,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._driver is None:
            self.connect()


def _truncate(text: str, max_len: int = 80) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text
