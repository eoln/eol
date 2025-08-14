"""
Centralized NetworkX mock for consistent test behavior across all test files.
This prevents test contamination from different mock implementations.
"""

import importlib.machinery
from unittest.mock import MagicMock


class MockMultiDiGraph:
    """Mock implementation of NetworkX MultiDiGraph for testing."""

    def __init__(self):
        self._nodes = {}
        self._edges = []
        self.nodes = self._nodes

    def __len__(self):
        """Return the number of nodes (NetworkX convention)."""
        return len(self._nodes)

    def __contains__(self, node_id):
        """Check if node exists in graph."""
        return node_id in self._nodes

    def __iter__(self):
        """Iterate over nodes."""
        return iter(self._nodes)

    def add_node(self, node_id, **attrs):
        self._nodes[node_id] = attrs

    def add_edge(self, source, target, **attrs):
        self._edges.append((source, target, attrs))
        # Ensure both nodes exist
        if source not in self._nodes:
            self._nodes[source] = {}
        if target not in self._nodes:
            self._nodes[target] = {}

    def has_node(self, node_id):
        return node_id in self._nodes

    def has_edge(self, source, target):
        return any(e[0] == source and e[1] == target for e in self._edges)

    def remove_node(self, node_id):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._edges = [
                (s, t, a) for s, t, a in self._edges if s != node_id and t != node_id
            ]

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def neighbors(self, node_id):
        return [e[1] for e in self._edges if e[0] == node_id]

    def degree(self, node_id=None):
        if node_id is None:
            # Return iterator of (node, degree) pairs for all nodes
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[0] == n or e[1] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            # Return degree for specific node
            return sum(1 for e in self._edges if e[0] == node_id or e[1] == node_id)

    def edges(self, data=False):
        """Return edges, optionally with data."""
        if data:
            return [(e[0], e[1], e[2]) for e in self._edges]
        else:
            return [(e[0], e[1]) for e in self._edges]

    def in_degree(self, node_id=None):
        """Return in-degree of node(s)."""
        if node_id is None:
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[1] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            return sum(1 for e in self._edges if e[1] == node_id)

    def out_degree(self, node_id=None):
        """Return out-degree of node(s)."""
        if node_id is None:
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[0] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            return sum(1 for e in self._edges if e[0] == node_id)


def create_networkx_mock():
    """Create a consistent NetworkX mock with proper spec."""
    nx_mock = MagicMock()
    nx_mock.__spec__ = importlib.machinery.ModuleSpec("networkx", None)
    nx_mock.MultiDiGraph = MockMultiDiGraph
    nx_mock.shortest_path = MagicMock(return_value=["node1", "node2", "node3"])
    return nx_mock
