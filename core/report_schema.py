from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel
from core.enums import AgentStatus

class EvidenceItem(BaseModel):
    role: Literal["Supports", "Contradicts", "Neutral", "Mixed"]
    severity: Optional[Literal["HIGH", "MEDIUM", "LOW"]]
    text: str
    source: str
    url: Optional[str] = None
    year: Optional[str] = None
    origin: Optional[str] = None
    
class EvidenceRoleCount(BaseModel):
    supports: int = 0
    contradicts: int = 0
    neutral: int = 0
    mixed: int = 0

class PeerEntry(BaseModel):
    company: str
    esg: Optional[float] = None
    greenwashing_risk_score: Optional[float] = None
    rank: Optional[str] = None
    data_source: str
    e: Optional[float] = None
    s: Optional[float] = None
    g: Optional[float] = None
    rating: Optional[str] = None
    is_target: Optional[bool] = False

class CredibilityTierCount(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0

class FactGraphSummary(BaseModel):
    nodes: int = 0
    edges: int = 0
    key_insights: List[str] = []

class NewsItem(BaseModel):
    headline: str
    source: str
    date: str
    sentiment: str

class ClaimIntensityResult(BaseModel):
    score: float
    status: AgentStatus
    fallback_reason: Optional[str] = None
    error: Optional[str] = None

class ControversyResult(BaseModel):
    score: float
    status: AgentStatus
    fallback_reason: Optional[str] = None
    error: Optional[str] = None

class TemporalResult(BaseModel):
    score: float
    status: AgentStatus
    fallback_reason: Optional[str] = None
    error: Optional[str] = None

class ReportPayload(BaseModel):
    # GW formula inputs
    claim_intensity: Optional[ClaimIntensityResult] = None
    performance_score: Optional[float] = None
    controversy_risk: Optional[ControversyResult] = None
    disclosure_score: Optional[float] = None
    temporal_escalation: Optional[TemporalResult] = None
    gw_formula_weights: Dict[str, float] = {}
    gw_score: Optional[float] = None
    esg_score: Optional[float] = None
    esg_lineage: Dict[str, Any] = {}

    # Evidence (single source of truth)
    unified_evidence: List[EvidenceItem] = []
    evidence_roles: EvidenceRoleCount = EvidenceRoleCount()

    # Peers
    peer_table: Optional[List[PeerEntry]] = None

    # NLP / Sentiment
    gsi_score: Optional[float] = None
    linguistic_risk: Optional[float] = None
    boilerplate_pct: Optional[float] = None
    sentiment_divergence: Optional[float] = None

    # Credibility
    avg_credibility: Optional[float] = None
    credibility_tier_counts: Optional[CredibilityTierCount] = None

    # Temporal conflict
    temporal_risk_level: Optional[Literal["LOW", "MODERATE", "HIGH"]] = None
    carbon_pathway_risk_level: Optional[Literal["LOW", "MODERATE", "HIGH"]] = None

    # Pipeline status
    pipeline_agent_statuses: Dict[str, AgentStatus] = {}

    # Adversarial / Fact graph
    adversarial_risk_score: Optional[float] = None
    adversarial_failed_agents: Optional[int] = None
    fact_graph_summary: Optional[FactGraphSummary] = None

    # Real-time news
    recent_news: Optional[List[NewsItem]] = None
    blocked_news_sources: Optional[List[str]] = None
