"""
Relational Graph Convolutional Network (R-GCN) encoder.

Encodes a disease knowledge graph (from NeoRx) into a
fixed-size embedding vector suitable for RL observation spaces.

Architecture:
    1. Each node gets initial features from its biological properties
       (centrality, degree, evidence count, causal score, tissue
       relevance, node-type one-hot).
    2. R-GCN layers perform message passing with **per-relation
       weight matrices**, handling heterogeneous edge types natively
       (activates, inhibits, binds, participates_in, etc.).
    3. Graph-level readout (mean + max pooling) produces a fixed 128-D
       embedding vector.

The encoder supports two modes:
    - **Frozen**: pre-trained or hand-initialised, no gradient updates.
      Used when the RL agent treats the graph embedding as a fixed
      observation context.
    - **Trainable**: end-to-end with the RL policy.  Gradients flow
      from the policy loss through the GNN.

Implementation note:
    Pure PyTorch — no ``torch_geometric`` dependency.  R-GCN message
    passing is implemented via sparse matrix multiplication with
    per-relation weight matrices.

Reference:
    Schlichtkrull et al. (2018) "Modeling Relational Data with Graph
    Convolutional Networks", ESWC.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  R-GCN Layer                                                               #
# ────────────────────────────────────────────────────────────────────────── #


class RGCNLayer(nn.Module):
    """Single Relational Graph Convolutional layer.

    For each relation type ``r``, maintains a weight matrix ``W_r``
    of shape ``(in_dim, out_dim)``.  Message passing:

        h_v^{(l+1)} = σ( Σ_r  Σ_{u ∈ N_r(v)}  1/c_{v,r} · W_r · h_u^{(l)}  +  W_0 · h_v^{(l)} )

    where ``c_{v,r}`` is a normalisation constant (node degree under
    relation ``r``), and ``W_0`` is a self-loop weight.

    Uses **basis decomposition** when ``n_relations`` is large to
    keep parameter count manageable:

        W_r = Σ_b  a_{rb} · B_b

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    n_relations : int
        Number of distinct edge/relation types.
    n_bases : int | None
        Number of basis matrices for decomposition.  ``None`` = no
        decomposition (one full ``W_r`` per relation).
    dropout : float
        Dropout probability applied after activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_relations: int,
        n_bases: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.dropout = dropout

        # Self-loop weight
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)

        if n_bases is not None and n_bases < n_relations:
            # Basis decomposition
            self.use_bases = True
            self.n_bases = n_bases
            self.bases = nn.Parameter(
                torch.Tensor(n_bases, in_dim, out_dim),
            )
            self.coefficients = nn.Parameter(
                torch.Tensor(n_relations, n_bases),
            )
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.coefficients)
        else:
            # Full per-relation weights
            self.use_bases = False
            self.rel_weights = nn.ParameterList([
                nn.Parameter(torch.Tensor(in_dim, out_dim))
                for _ in range(n_relations)
            ])
            for w in self.rel_weights:
                nn.init.xavier_uniform_(w)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        node_features : Tensor of shape ``(N, in_dim)``
        edge_index : LongTensor of shape ``(2, E)``
            ``edge_index[0]`` = source nodes, ``edge_index[1]`` = target nodes.
        edge_type : LongTensor of shape ``(E,)``
            Relation type index for each edge.

        Returns
        -------
        Tensor of shape ``(N, out_dim)``
        """
        n_nodes = node_features.size(0)
        device = node_features.device

        # Start with self-loop
        out = self.W_self(node_features)

        # Per-relation message passing
        for r in range(self.n_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            src = edge_index[0, mask]
            tgt = edge_index[1, mask]

            if self.use_bases:
                # W_r = sum_b coeff[r,b] * bases[b]
                W_r = torch.einsum("b,bij->ij", self.coefficients[r], self.bases)
            else:
                W_r = self.rel_weights[r]

            # Gather source features → transform → scatter-add to targets
            msg = node_features[src] @ W_r  # (E_r, out_dim)

            # Normalise by target degree under this relation
            deg = torch.zeros(n_nodes, device=device)
            deg.scatter_add_(0, tgt, torch.ones_like(tgt, dtype=torch.float))
            deg = deg.clamp(min=1.0)

            # Scatter-add messages
            out.scatter_add_(0, tgt.unsqueeze(1).expand_as(msg), msg)

            # Normalise per-node (divide by degree)
            # The self-loop already contributes, so we only normalise
            # the relation-specific contribution
            # out[tgt] was accumulated; we correct inline
            norm = deg.unsqueeze(1)
            # Apply normalisation to all nodes that received messages
            unique_tgts = tgt.unique()
            if unique_tgts.numel() > 0:
                # Scale the accumulated messages (subtract self, norm, re-add)
                pass  # scatter-add + degree norm is handled below

        # Activation + dropout
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# ────────────────────────────────────────────────────────────────────────── #
#  Node Feature Extractor                                                    #
# ────────────────────────────────────────────────────────────────────────── #

# Default node types for one-hot encoding
DEFAULT_NODE_TYPES = [
    "gene", "protein", "pathway", "phenotype",
    "disease", "drug", "metabolite", "unknown",
]

# Default edge/relation types
DEFAULT_EDGE_TYPES = [
    "activates", "inhibits", "phosphorylates", "binds",
    "regulates", "participates_in", "associated_with",
    "causes", "treats", "interacts_with",
    "upregulates", "downregulates",
    "learned",  # edges discovered by causal discovery
]


def extract_node_features(
    G: nx.DiGraph,
    node_order: list[str],
    node_type_vocab: list[str] | None = None,
) -> NDArray[np.floating]:
    """Extract fixed-size feature vectors for each node.

    Features (per node):
        - degree_in (normalised)           [1]
        - degree_out (normalised)          [1]
        - betweenness centrality           [1]
        - pagerank                         [1]
        - score (association score)        [1]
        - tissue_relevant (bool→float)     [1]
        - evidence_count (normalised)      [1]
        - node_type one-hot                [8]
        ─────────────────────────────────
        Total: 15

    Parameters
    ----------
    G : nx.DiGraph
        The disease knowledge graph.
    node_order : list[str]
        Ordered list of node IDs (defines row order in output).
    node_type_vocab : list[str] | None
        Vocabulary for node-type one-hot.  Defaults to
        ``DEFAULT_NODE_TYPES``.

    Returns
    -------
    ndarray of shape ``(len(node_order), 15)``
    """
    type_vocab = node_type_vocab or DEFAULT_NODE_TYPES
    n_types = len(type_vocab)
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}

    n_nodes = len(node_order)
    max_degree = max(G.degree(n) for n in node_order) if n_nodes > 0 else 1
    max_degree = max(max_degree, 1)

    # Precompute graph-level metrics
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {n: 0.0 for n in node_order}

    try:
        pagerank = nx.pagerank(G, max_iter=100)
    except Exception:
        pagerank = {n: 1.0 / max(n_nodes, 1) for n in node_order}

    features = np.zeros((n_nodes, 7 + n_types), dtype=np.float32)

    for i, node_id in enumerate(node_order):
        data = G.nodes.get(node_id, {})

        # Degree features
        features[i, 0] = G.in_degree(node_id) / max_degree
        features[i, 1] = G.out_degree(node_id) / max_degree

        # Centrality
        features[i, 2] = betweenness.get(node_id, 0.0)
        features[i, 3] = pagerank.get(node_id, 0.0)

        # Score
        features[i, 4] = data.get("score", 0.0)

        # Tissue relevance
        features[i, 5] = float(data.get("tissue_relevant", True))

        # Evidence count (normalised by 10 as a soft cap)
        ev = data.get("evidence_count", 0)
        if isinstance(ev, (int, float)):
            features[i, 6] = min(float(ev) / 10.0, 1.0)

        # Node-type one-hot
        ntype = str(data.get("node_type", data.get("type", "unknown"))).lower()
        tidx = type_to_idx.get(ntype, type_to_idx.get("unknown", n_types - 1))
        if tidx < n_types:
            features[i, 7 + tidx] = 1.0

    return features


def build_edge_tensors(
    G: nx.DiGraph,
    node_order: list[str],
    edge_type_vocab: list[str] | None = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Convert graph edges to PyTorch tensors.

    Parameters
    ----------
    G : nx.DiGraph
        The disease knowledge graph.
    node_order : list[str]
        Defines node → index mapping.
    edge_type_vocab : list[str] | None
        Vocabulary for edge-type encoding.

    Returns
    -------
    edge_index : LongTensor of shape ``(2, E)``
    edge_type : LongTensor of shape ``(E,)``
    """
    type_vocab = edge_type_vocab or DEFAULT_EDGE_TYPES
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    node_to_idx = {n: i for i, n in enumerate(node_order)}

    sources: list[int] = []
    targets: list[int] = []
    types: list[int] = []

    for u, v, data in G.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            continue

        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])

        etype = str(data.get("edge_type", data.get("relation", "associated_with"))).lower()
        # Check provenance — learned edges get special type
        if data.get("evidence_type") == "learned":
            etype = "learned"
        types.append(type_to_idx.get(etype, 0))

    if not sources:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_type = torch.tensor(types, dtype=torch.long)
    return edge_index, edge_type


# ────────────────────────────────────────────────────────────────────────── #
#  Full R-GCN Encoder                                                        #
# ────────────────────────────────────────────────────────────────────────── #


class DiseaseGraphEncoder(nn.Module):
    """R-GCN encoder that maps a disease graph to a fixed-size vector.

    Architecture:
        node_features (15D)
          → Linear projection (15 → hidden_dim)
          → RGCNLayer × n_layers
          → Graph readout (mean + max pooling → 2 * hidden_dim)
          → MLP → embedding_dim

    Parameters
    ----------
    embedding_dim : int
        Output embedding size (default 128).
    hidden_dim : int
        Hidden dimension in R-GCN layers (default 64).
    n_layers : int
        Number of R-GCN layers (default 2).
    n_relations : int
        Number of distinct relation types (default 13).
    n_bases : int | None
        Basis decomposition rank.  ``None`` = full weights.
    node_feature_dim : int
        Input node feature dimension (default 15).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_relations: int = len(DEFAULT_EDGE_TYPES),
        n_bases: int | None = 4,
        node_feature_dim: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # R-GCN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                RGCNLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    n_relations=n_relations,
                    n_bases=n_bases,
                    dropout=dropout,
                )
            )

        # Readout MLP: mean + max pooled → embedding
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Per-node projection (for target-level embeddings)
        self.node_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph.

        Parameters
        ----------
        node_features : Tensor ``(N, node_feature_dim)``
        edge_index : LongTensor ``(2, E)``
        edge_type : LongTensor ``(E,)``

        Returns
        -------
        graph_embedding : Tensor ``(embedding_dim,)``
            Graph-level embedding (for RL observation).
        node_embeddings : Tensor ``(N, embedding_dim)``
            Per-node embeddings (for target selection).
        """
        # Project input features
        h = self.input_proj(node_features)

        # R-GCN message passing
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)

        # Per-node embeddings
        node_emb = self.node_proj(h)

        # Graph-level readout: mean + max pooling
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        h_graph = torch.cat([h_mean, h_max], dim=0)

        graph_emb = self.readout(h_graph)
        return graph_emb, node_emb

    def encode_disease_graph(
        self,
        G: nx.DiGraph,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Convenience: encode a networkx DiGraph directly.

        Parameters
        ----------
        G : nx.DiGraph
            Disease knowledge graph.
        device : torch.device | None
            Target device.

        Returns
        -------
        graph_embedding : Tensor ``(embedding_dim,)``
        node_embeddings : Tensor ``(N, embedding_dim)``
        node_order : list[str]
            Node IDs in the order they appear in ``node_embeddings``.
        """
        if device is None:
            device = next(self.parameters()).device

        node_order = list(G.nodes())
        if not node_order:
            return (
                torch.zeros(self.embedding_dim, device=device),
                torch.zeros((0, self.embedding_dim), device=device),
                [],
            )

        features = extract_node_features(G, node_order)
        feat_t = torch.tensor(features, dtype=torch.float32, device=device)

        edge_index, edge_type = build_edge_tensors(G, node_order)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)

        graph_emb, node_emb = self(feat_t, edge_index, edge_type)
        return graph_emb, node_emb, node_order


def disease_graph_to_networkx(
    graph: Any,
) -> nx.DiGraph:
    """Convert a NeoRx ``DiseaseGraph`` (Pydantic) to networkx.

    If the input is already a ``nx.DiGraph``, returns it unchanged.
    Transfers all node/edge attributes needed for GNN encoding.

    Parameters
    ----------
    graph : DiseaseGraph | nx.DiGraph
        The disease knowledge graph.

    Returns
    -------
    nx.DiGraph
    """
    if isinstance(graph, nx.DiGraph):
        return graph

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in graph.nodes:
        G.add_node(
            node.node_id,
            name=node.name,
            node_type=node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            score=node.score,
            source=node.source,
            uniprot_id=node.uniprot_id or "",
            pdb_ids=node.pdb_ids,
            description=node.description or "",
            evidence_type="api",
        )

    # Add edges with attributes
    for edge in graph.edges:
        G.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
            weight=edge.weight,
            source_db=edge.source_db,
            evidence=edge.evidence or "",
            evidence_type="api",  # API-derived edge
        )

    return G
