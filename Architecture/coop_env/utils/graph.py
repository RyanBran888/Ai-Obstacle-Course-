"""An undirected graph over hashable nodes, used for the region topology.

Every traversal here returns results in a deterministic order so that two runs
with the same seed produce byte-identical rooms.
"""

from __future__ import annotations

from collections import deque
from typing import Hashable, Iterable, Iterator, TypeVar

Node = TypeVar("Node", bound=Hashable)


class Graph[Node: Hashable]:
    """Undirected adjacency-set graph."""

    __slots__ = ("_adj",)

    def __init__(self) -> None:
        self._adj: dict[Node, set[Node]] = {}

    # -- construction ----------------------------------------------------

    def add_node(self, node: Node) -> None:
        self._adj.setdefault(node, set())

    def add_edge(self, a: Node, b: Node) -> None:
        if a == b:
            return
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)

    def remove_edge(self, a: Node, b: Node) -> None:
        self._adj.get(a, set()).discard(b)
        self._adj.get(b, set()).discard(a)

    # -- queries ---------------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        return sorted(self._adj)

    def neighbors(self, node: Node) -> list[Node]:
        return sorted(self._adj.get(node, ()))

    def degree(self, node: Node) -> int:
        return len(self._adj.get(node, ()))

    def has_edge(self, a: Node, b: Node) -> bool:
        return b in self._adj.get(a, ())

    def edges(self) -> list[tuple[Node, Node]]:
        seen: set[tuple[Node, Node]] = set()
        for a in sorted(self._adj):
            for b in sorted(self._adj[a]):
                key = (a, b) if a <= b else (b, a)  # type: ignore[operator]
                seen.add(key)
        return sorted(seen)

    def __contains__(self, node: object) -> bool:
        return node in self._adj

    def __len__(self) -> int:
        return len(self._adj)

    def __iter__(self) -> Iterator[Node]:
        return iter(sorted(self._adj))

    # -- traversal -------------------------------------------------------

    def bfs_order(self, start: Node) -> list[Node]:
        if start not in self._adj:
            return []
        order = [start]
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for n in sorted(self._adj[current]):
                if n not in seen:
                    seen.add(n)
                    order.append(n)
                    queue.append(n)
        return order

    def depths(self, start: Node) -> dict[Node, int]:
        """Edge-count distance from `start` to every reachable node."""
        if start not in self._adj:
            return {}
        depth = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for n in sorted(self._adj[current]):
                if n not in depth:
                    depth[n] = depth[current] + 1
                    queue.append(n)
        return depth

    def bfs_tree(self, start: Node) -> dict[Node, Node | None]:
        """Parent map of the breadth-first spanning tree rooted at `start`."""
        if start not in self._adj:
            return {}
        parent: dict[Node, Node | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for n in sorted(self._adj[current]):
                if n not in parent:
                    parent[n] = current
                    queue.append(n)
        return parent

    def path_to_root(self, node: Node, parent: dict[Node, Node | None]) -> list[Node]:
        """Chain of nodes from `node` up to the tree root, inclusive."""
        chain = [node]
        current = node
        while parent.get(current) is not None:
            current = parent[current]  # type: ignore[assignment]
            chain.append(current)
        return chain

    def components(self) -> list[list[Node]]:
        seen: set[Node] = set()
        result: list[list[Node]] = []
        for node in sorted(self._adj):
            if node in seen:
                continue
            group = self.bfs_order(node)
            seen.update(group)
            result.append(sorted(group))
        return result

    def is_connected(self) -> bool:
        return len(self._adj) == 0 or len(self.components()) == 1

    def reachable_from(
        self, start: Node, blocked_edges: Iterable[tuple[Node, Node]] = ()
    ) -> set[Node]:
        """Nodes reachable from `start` while pretending `blocked_edges` are absent."""
        blocked: set[frozenset[Node]] = {frozenset(e) for e in blocked_edges}
        if start not in self._adj:
            return set()
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for n in sorted(self._adj[current]):
                if n in seen or frozenset((current, n)) in blocked:
                    continue
                seen.add(n)
                queue.append(n)
        return seen

    def subgraph(self, nodes: Iterable[Node]) -> "Graph[Node]":
        keep = set(nodes)
        sub: Graph[Node] = Graph()
        for node in keep:
            sub.add_node(node)
            for n in self._adj.get(node, ()):
                if n in keep:
                    sub.add_edge(node, n)
        return sub
