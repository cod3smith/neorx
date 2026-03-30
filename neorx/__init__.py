"""
NeoRx — Public API
===========================

Thin re-export package so users can write clean imports::

    from neorx import run_pipeline, build_disease_graph
    from neorx import identify_causal_targets
    from neorx import predict_admet, save_graph

Instead of the internal layout::

    from modules.neorx import run_pipeline   # also works
"""

from modules.neorx import (  # noqa: F401
    # Enums
    NodeType,
    EdgeType,
    JobStatus,
    TargetClassification,
    # Graph models
    GraphNode,
    GraphEdge,
    DiseaseGraph,
    # Analysis models
    NeoRxResult,
    ScoredCandidate,
    # Pipeline models
    PipelineJob,
    PipelineResult,
    # API models
    RunRequest,
    GraphRequest,
    IdentifyRequest,
    ScreenTargetRequest,
    StatusResponse,
    # Core functions
    build_disease_graph,
    disease_graph_to_networkx,
    identify_causal_targets,
    score_candidate,
    rank_candidates,
    normalise_affinity,
    normalise_sa,
    run_pipeline,
    generate_report,
    # Cache
    get_cache,
    cached_api_call,
    store_api_response,
    # Persistence
    save_graph,
    load_graph,
    list_saved_graphs,
    # ADMET
    predict_admet,
    ADMETProfile,
)

__version__ = "0.1.0"

__all__ = [
    # Enums
    "NodeType",
    "EdgeType",
    "JobStatus",
    "TargetClassification",
    # Graph models
    "GraphNode",
    "GraphEdge",
    "DiseaseGraph",
    # Analysis models
    "NeoRxResult",
    "ScoredCandidate",
    # Pipeline models
    "PipelineJob",
    "PipelineResult",
    # API models
    "RunRequest",
    "GraphRequest",
    "IdentifyRequest",
    "ScreenTargetRequest",
    "StatusResponse",
    # Core functions
    "build_disease_graph",
    "disease_graph_to_networkx",
    "identify_causal_targets",
    "score_candidate",
    "rank_candidates",
    "normalise_affinity",
    "normalise_sa",
    "run_pipeline",
    "generate_report",
    # Cache
    "get_cache",
    "cached_api_call",
    "store_api_response",
    # Persistence
    "save_graph",
    "load_graph",
    "list_saved_graphs",
    # ADMET
    "predict_admet",
    "ADMETProfile",
]
