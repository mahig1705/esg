"""
Strict domain schema for the company-centric ESG knowledge graph.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


ENTITY_TYPES = {
    "Organization",
    "KPI",
    "Facility",
    "RegulatoryVerdict",
    "SustainabilityGoal",
    "EvidenceSource",
}

RELATIONSHIP_TYPES = {
    "HAS_KPI",
    "HAS_REGULATORY_VERDICT",
    "HAS_SUSTAINABILITY_GOAL",
    "HAS_FACILITY",
    "SUPPORTED_BY",
}

ALLOWED_RELATIONSHIPS = {
    ("Organization", "HAS_KPI", "KPI"),
    ("Organization", "HAS_REGULATORY_VERDICT", "RegulatoryVerdict"),
    ("Organization", "HAS_SUSTAINABILITY_GOAL", "SustainabilityGoal"),
    ("Organization", "HAS_FACILITY", "Facility"),
    ("Facility", "HAS_REGULATORY_VERDICT", "RegulatoryVerdict"),
    ("Facility", "HAS_KPI", "KPI"),
    ("KPI", "SUPPORTED_BY", "EvidenceSource"),
    ("RegulatoryVerdict", "SUPPORTED_BY", "EvidenceSource"),
    ("SustainabilityGoal", "SUPPORTED_BY", "EvidenceSource"),
    ("Facility", "SUPPORTED_BY", "EvidenceSource"),
}

MANDATORY_MEASURE_PROPERTIES = {
    "value": float,
    "unit": str,
    "year": int,
    "source_tier": int,
}


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid float metric value.")
    return float(value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid integer metric value.")
    return int(value)


def validate_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entity, dict):
        raise TypeError("Entity must be a dict.")

    label = str(entity.get("label") or "").strip()
    if label not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity label: {label}")

    node_key = str(entity.get("node_key") or "").strip()
    if not node_key:
        raise ValueError(f"Entity '{label}' missing node_key.")

    properties = entity.get("properties", {})
    if not isinstance(properties, dict):
        raise TypeError(f"Entity '{label}' properties must be a dict.")

    if label in {"KPI", "SustainabilityGoal"}:
        for key, expected_type in MANDATORY_MEASURE_PROPERTIES.items():
            if key not in properties:
                raise ValueError(f"Entity '{label}' missing mandatory property '{key}'.")
            if expected_type is float:
                properties[key] = _coerce_float(properties[key])
            elif expected_type is int:
                properties[key] = _coerce_int(properties[key])
            else:
                properties[key] = str(properties[key])

    entity["label"] = label
    entity["node_key"] = node_key
    entity["properties"] = properties
    return entity


def validate_relationship(relationship: Dict[str, Any], entity_lookup: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(relationship, dict):
        raise TypeError("Relationship must be a dict.")

    source_key = str(relationship.get("source_key") or "").strip()
    target_key = str(relationship.get("target_key") or "").strip()
    rel_type = str(relationship.get("type") or "").strip()

    if not source_key or not target_key or not rel_type:
        raise ValueError("Relationship missing source_key, target_key, or type.")
    if rel_type not in RELATIONSHIP_TYPES:
        raise ValueError(f"Unsupported relationship type: {rel_type}")

    source_label = entity_lookup.get(source_key)
    target_label = entity_lookup.get(target_key)
    if not source_label or not target_label:
        raise ValueError(f"Relationship references unknown nodes: {source_key} -> {target_key}")
    if (source_label, rel_type, target_label) not in ALLOWED_RELATIONSHIPS:
        raise ValueError(
            f"Relationship not allowed by schema: ({source_label})-[:{rel_type}]->({target_label})"
        )

    relationship["source_key"] = source_key
    relationship["target_key"] = target_key
    relationship["type"] = rel_type
    return relationship


def validate_graph_package(
    organization: Dict[str, Any],
    entities: Iterable[Dict[str, Any]],
    relationships: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    org = validate_entity(organization)
    if org.get("label") != "Organization":
        raise ValueError("Graph package must be anchored on an Organization node.")

    cleaned_entities: List[Dict[str, Any]] = [org]
    for entity in entities:
        cleaned_entities.append(validate_entity(entity))

    lookup = {entity["node_key"]: entity["label"] for entity in cleaned_entities}
    cleaned_relationships = [validate_relationship(rel, lookup) for rel in relationships]

    return {
        "organization": org,
        "entities": cleaned_entities[1:],
        "relationships": cleaned_relationships,
    }
