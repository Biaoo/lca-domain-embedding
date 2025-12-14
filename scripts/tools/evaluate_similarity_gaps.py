"""
Evaluate how fine-tuned embedding models handle nuanced semantic differences.

This script focuses on scenario-based checks where a single query must prefer the
correct document among two similar-but-different contents. Each function targets
one type of distinction (region, data type, time, technology). For every case we
compare baseline vs. fine-tuned models and report the margin gain.

Usage example:
    uv run python scripts/evaluate_similarity_gaps.py \
        --model_pairs bge,qwen \
        --output_json data/eval/similarity_gap_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.cal_similarity import cosine_similarity
from src.embedding import get_embedding

MODEL_PAIRS: dict[str, dict[str, str]] = {
    "bge": {
        "name": "bge-m3",
        "baseline": "data/model/BAAI--bge-m3",
        "finetuned": "data/output/lca-bge-m3-finetuned",
    },
    "qwen": {
        "name": "qwen3",
        "baseline": "data/model/Qwen--Qwen3-Embedding-0.6B",
        "finetuned": "data/output/lca-qwen3-st-finetuned",
    },
}

VIEWER_TEMPLATE = ROOT_DIR / "docs" / "similarity_gap_viewer.html"
EMBED_PLACEHOLDER = '<script id="embedded-data" type="application/json"></script>'

METRIC_EXPLANATION_ZH = """
指标说明：
- Baseline/Finetuned → pos/contrast：同一个查询与目标文档及对照文档的余弦相似度，反映模型关注点。
- margin：pos − contrast，数值越大表示模型更好地区分正确文档与干扰文档。
- Δmargin (margin_gain)：微调后 margin − 微调前 margin，大于 0 代表辨识力提升。
- Δpos (positive_gain)：微调后与目标文档的相似度提升量。
- Δneg (contrast_gain)：微调后与对照文档的相似度变化，若为负值表示模型更好地抑制干扰。
- improvement_rate：在全部场景中，margin_gain > 0 的占比。
"""


@dataclass
class EvaluationCase:
    category: str
    description: str
    query: str
    positive_label: str
    contrast_label: str
    positive_doc: Path | None = None
    contrast_doc: Path | None = None
    positive_text: str | None = None
    contrast_text: str | None = None

    def __post_init__(self) -> None:
        if not self.positive_text:
            if self.positive_doc is None:
                raise ValueError("Positive doc path or text must be provided.")
            if not self.positive_doc.is_absolute():
                object.__setattr__(self, "positive_doc", (ROOT_DIR / self.positive_doc).resolve())
            if not self.positive_doc.exists():
                raise FileNotFoundError(f"Positive doc not found: {self.positive_doc}")
        if not self.contrast_text:
            if self.contrast_doc is None:
                raise ValueError("Contrast doc path or text must be provided.")
            if not self.contrast_doc.is_absolute():
                object.__setattr__(self, "contrast_doc", (ROOT_DIR / self.contrast_doc).resolve())
            if not self.contrast_doc.exists():
                raise FileNotFoundError(f"Contrast doc not found: {self.contrast_doc}")


EmbeddingCache = dict[tuple[str, str], list[float]]
_embedding_cache: EmbeddingCache = {}


def get_vector(text: str, model: str) -> list[float]:
    """Cache embeddings to avoid recomputation."""
    key = (model, text)
    if key not in _embedding_cache:
        emb = get_embedding(text, model=model)
        if emb is None:
            raise RuntimeError(f"Failed to encode text with model '{model}'")
        _embedding_cache[key] = emb
    return _embedding_cache[key]


@lru_cache(maxsize=None)
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def compute_similarity(query: str, document: str, model: str) -> float:
    query_vec = get_vector(query, model)
    doc_vec = get_vector(document, model)
    return float(cosine_similarity(query_vec, doc_vec))


def resolve_content(path: Path | None, inline_text: str | None, fallback_label: str) -> tuple[str, str]:
    if inline_text is not None:
        return fallback_label, inline_text.strip()
    if path is None:
        raise ValueError("Content requires either inline text or file path.")
    text = read_text(str(path))
    return str(path.relative_to(ROOT_DIR)), text


def evaluate_case(case: EvaluationCase, model_pair: dict[str, str]) -> dict[str, object]:
    """Evaluate a single scenario for one baseline/finetuned pair."""
    query = case.query.strip()
    positive_path, positive = resolve_content(case.positive_doc, case.positive_text, "[inline_positive]")
    contrast_path, contrast = resolve_content(case.contrast_doc, case.contrast_text, "[inline_contrast]")

    base_pos = compute_similarity(query, positive, model_pair["baseline"])
    base_neg = compute_similarity(query, contrast, model_pair["baseline"])
    ft_pos = compute_similarity(query, positive, model_pair["finetuned"])
    ft_neg = compute_similarity(query, contrast, model_pair["finetuned"])

    base_margin = base_pos - base_neg
    ft_margin = ft_pos - ft_neg

    return {
        "category": case.category,
        "description": case.description,
        "model_pair": model_pair["name"],
        "query": query,
        "positive_doc": positive_path,
        "contrast_doc": contrast_path,
        "labels": {
            "positive": case.positive_label,
            "contrast": case.contrast_label,
        },
        "doc_text": {
            "positive": positive,
            "contrast": contrast,
        },
        "baseline": {
            "positive": base_pos,
            "contrast": base_neg,
            "margin": base_margin,
        },
        "finetuned": {
            "positive": ft_pos,
            "contrast": ft_neg,
            "margin": ft_margin,
        },
        "margin_gain": ft_margin - base_margin,
        "positive_gain": ft_pos - base_pos,
        "contrast_gain": ft_neg - base_neg,
    }


def evaluate_region_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Region-focused scenario: global (GLO) vs China-specific (CN) coal datasets."""
    case = EvaluationCase(
        category="region_gap",
        description="Distinguish CN hard coal import logistics from the global market group.",
        query="hard coal import china",
        positive_doc=Path("data/markdown/c5e96c1b-1fe6-3df6-81a4-6813d2b7c56c.md"),
        contrast_doc=Path("data/markdown/df8a13bd-1696-348d-bdee-a5dd10837350.md"),
        positive_label="Import route to China (Location: CN)",
        contrast_label="Global market group (Location: GLO)",
    )
    return evaluate_case(case, model_pair)


def evaluate_data_type_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Data-type scenario: waste treatment vs manufacturing process."""
    case = EvaluationCase(
        category="data_type_gap",
        description="Scrap steel disposal in MSWI vs. downstream wire drawing production.",
        query="steel waste treatment",
        positive_doc=Path("data/markdown/c4acfc12-b703-3641-9177-12f27b365683.md"),
        contrast_doc=Path("data/markdown/dd0a1e98-1426-3b38-84ba-6f1431bba2f6.md"),
        positive_label="Waste treatment dataset (incineration FAE)",
        contrast_label="Wire drawing production step",
    )
    return evaluate_case(case, model_pair)


def evaluate_time_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Temporal scenario: modern (2017) vs legacy (1997) hot rolling operations."""
    case = EvaluationCase(
        category="time_gap",
        description="Modern Austrian hot rolling (2017) vs older EU average (1997).",
        query="steel production austria 2017",
        positive_doc=Path("data/markdown/b24376a9-177c-3984-b007-66ee79f183d1.md"),
        contrast_doc=Path("data/markdown/e67df727-8046-31af-8828-b3684c6f568a.md"),
        positive_label="Austria-specific 2017 dataset",
        contrast_label="RoW / EU average 1997 dataset",
    )
    return evaluate_case(case, model_pair)


def evaluate_technology_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Technology scenario: Electric arc furnace vs basic oxygen furnace steel routes."""
    case = EvaluationCase(
        category="technology_gap",
        description="Electric arc furnace reinforcing steel vs BOF converter route.",
        query="steel production electric arc furnace",
        positive_doc=Path("data/markdown/8a7b1681-54db-3dfa-a14f-e8d3d9a6ebc9.md"),
        contrast_doc=Path("data/markdown/2bc4178d-827f-3334-9f09-0d29d32812cf.md"),
        positive_label="Secondary steel via EAF",
        contrast_label="Primary steel via BOF",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Location preference scenario resembling visualize_similarity examples."""
    case = EvaluationCase(
        category="location_gap",
        description="Location-aware ranking: Austria dataset versus global mix.",
        query="steel production in austria",
        positive_doc=Path("data/markdown/b24376a9-177c-3984-b007-66ee79f183d1.md"),
        contrast_doc=Path("data/markdown/9ed5f062-bef6-309e-b265-80773cf14d82.md"),
        positive_label="Hot rolling (Location: AT)",
        contrast_label="Hot rolling mix (Location: RoW)",
    )
    return evaluate_case(case, model_pair)


def evaluate_disposal_method_difference(model_pair: dict[str, str]) -> dict[str, object]:
    """Compare landfill vs incineration treatments for the same waste type."""
    case = EvaluationCase(
        category="disposal_gap",
        description="Sanitary landfill vs MSWI incineration for untreated waste wood.",
        query="waste wood disposal sanitary landfill switzerland",
        positive_doc=Path("data/markdown/3c723bbd-166e-370f-9f23-13a8f3234923.md"),
        contrast_doc=Path("data/markdown/ba731eb5-1105-30d3-9c80-8ecfefb7e5f9.md"),
        positive_label="Sanitary landfill (Location: CH)",
        contrast_label="MSWI incineration (Location: CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_market_vs_process(model_pair: dict[str, str]) -> dict[str, object]:
    """Differentiate a specific generation dataset from a global market."""
    case = EvaluationCase(
        category="market_gap",
        description="Run-of-river hydropower plant in Assam vs global hydropower market.",
        query="run-of-river hydropower electricity in assam india",
        positive_doc=Path("data/markdown/76d161a2-ec07-3f16-a585-7129d3b57f0f.md"),
        contrast_doc=Path("data/markdown/09d9b222-46c3-326b-b364-25675925a051.md"),
        positive_label="Hydropower plant (Location: IN-AS)",
        contrast_label="Global market for run-of-river plants",
    )
    return evaluate_case(case, model_pair)


def evaluate_attribute_mix_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Check recognition of attribute-mix vs fossil-specific electricity."""
    case = EvaluationCase(
        category="attribute_mix_gap",
        description="European attribute electricity mix vs Xinjiang natural gas plant.",
        query="european attribute mix medium voltage electricity 2023",
        positive_doc=Path("data/markdown/ef440aff-8fbf-3987-bb33-bb93468bcd0d.md"),
        contrast_doc=Path("data/markdown/421e4db7-5dc0-35ce-962a-dcddc6072f8f.md"),
        positive_label="European attribute mix (Medium voltage)",
        contrast_label="Natural gas plant (Location: CN-XJ)",
    )
    return evaluate_case(case, model_pair)


def evaluate_hazard_vs_nonhazard(model_pair: dict[str, str]) -> dict[str, object]:
    """Hazardous landfill vs sanitary landfill classification."""
    case = EvaluationCase(
        category="hazard_gap",
        description="Hazardous nickel slag landfill versus sanitary landfill of untreated wood.",
        query="hazardous waste landfill nickel slag switzerland",
        positive_doc=Path("data/markdown/e67ec1f6-a694-3a98-bd5a-f2164045063b.md"),
        contrast_doc=Path("data/markdown/3c723bbd-166e-370f-9f23-13a8f3234923.md"),
        positive_label="Residual material landfill (Hazardous, CH)",
        contrast_label="Sanitary landfill for waste wood (CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_supply_chain_stage(model_pair: dict[str, str]) -> dict[str, object]:
    """Differentiate upstream mining vs downstream refining of aluminium."""
    case = EvaluationCase(
        category="supply_chain_gap",
        description="Aluminium oxide refining in EU vs global bauxite mining upstream.",
        query="aluminium oxide refining eu27",
        positive_doc=Path("data/markdown/73c17366-0133-32c5-b96b-d61cc4ed6b1a.md"),
        contrast_doc=Path("data/markdown/7b8cd7f2-7112-36c0-97df-3556c92da275.md"),
        positive_label="Aluminium oxide production (EU27 & EFTA)",
        contrast_label="Bauxite mine operation (Global)",
    )
    return evaluate_case(case, model_pair)


def evaluate_energy_source_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Renewable (wind) vs fossil (coal) electricity in the same region."""
    case = EvaluationCase(
        category="energy_source_gap",
        description="Onshore wind (Ningxia) vs hard coal electricity (China) for the same region.",
        query="onshore wind electricity production ningxia",
        positive_doc=Path("data/markdown/e4597fba-bb41-3618-afc6-8b619eac92b1.md"),
        contrast_doc=Path("data/markdown/a2b3cf68-fa93-395a-9073-31c5b87dd06a.md"),
        positive_label="Wind power 1-3MW (Location: CN-NX)",
        contrast_label="Hard coal power plant (Location: CN)",
    )
    return evaluate_case(case, model_pair)


def evaluate_circularity_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Virgin vs recycled newsprint production."""
    case = EvaluationCase(
        category="circularity_gap",
        description="Newsprint recycled in Switzerland vs virgin newsprint.",
        query="recycled newsprint production switzerland",
        positive_doc=Path("data/markdown/b3fc5c97-ccd2-3eba-b452-439f175243dd.md"),
        contrast_doc=Path("data/markdown/160bc751-5804-32e3-a072-d4c075a9688f.md"),
        positive_label="Newsprint, recycled (Location: CH)",
        contrast_label="Newsprint, virgin (Location: CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_aluminium_chain_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Primary ingot vs alloy route heavy on scrap."""
    case = EvaluationCase(
        category="aluminium_chain_gap",
        description="Primary aluminium ingot vs scrap-intensive aluminium alloy.",
        query="primary aluminium ingot eu27 production",
        positive_doc=Path("data/markdown/bf80ed6c-bde7-3d52-abae-f0bc3cdc9213.md"),
        contrast_doc=Path("data/markdown/77ed64f5-3a70-3723-80d2-c44bc1af8edb.md"),
        positive_label="Primary aluminium ingot (EU27 & EFTA)",
        contrast_label="AlLi alloy (scrap intensive mix)",
    )
    return evaluate_case(case, model_pair)


def evaluate_waste_market_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Waste collection market vs actual treatment."""
    case = EvaluationCase(
        category="waste_market_gap",
        description="Waste wood collection market in Austria vs its incineration treatment.",
        query="waste wood collection market austria",
        positive_doc=Path("data/markdown/dadb1730-3161-34ff-a8dc-0be8007e76f0.md"),
        contrast_doc=Path("data/markdown/ba731eb5-1105-30d3-9c80-8ecfefb7e5f9.md"),
        positive_label="Market for waste wood, untreated (AT)",
        contrast_label="MSWI incineration of waste wood (CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_infrastructure_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Construction/infrastructure vs electricity generation."""
    case = EvaluationCase(
        category="infrastructure_gap",
        description="Wind turbine construction inventory vs wind electricity production.",
        query="wind turbine construction materials 2mw onshore",
        positive_doc=Path("data/markdown/aecd79c8-dfed-3db5-8146-fe36471d004f.md"),
        contrast_doc=Path("data/markdown/e4597fba-bb41-3618-afc6-8b619eac92b1.md"),
        positive_label="Wind turbine construction (2MW onshore)",
        contrast_label="Wind electricity production (CN-NX)",
    )
    return evaluate_case(case, model_pair)


def evaluate_maintenance_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Maintenance activity vs equipment manufacturing."""
    case = EvaluationCase(
        category="maintenance_gap",
        description="Pellet Stirling CHP maintenance vs pellet furnace manufacturing.",
        query="pellet stirling maintenance schedule",
        positive_doc=Path("data/markdown/dd452da1-acb0-3614-bbe0-c6998a41ceff.md"),
        contrast_doc=Path("data/markdown/6a6f6974-0800-3faa-86ae-35a413879497.md"),
        positive_label="Maintenance of pellet CHP (RoW)",
        contrast_label="Pellet furnace production (CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_clinker_market_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Manufacturing dataset vs market mix in construction materials."""
    case = EvaluationCase(
        category="clinker_market_gap",
        description="Swiss clinker production vs European quicklime market mix.",
        query="clinker production switzerland alternative fuels",
        positive_doc=Path("data/markdown/6df7c40e-f502-38ed-9837-b0ad816b4452.md"),
        contrast_doc=Path("data/markdown/4ca0dab5-185f-3bae-b26f-3adc6231b13e.md"),
        positive_label="Clinker production (CH)",
        contrast_label="Market for quicklime (RER)",
    )
    return evaluate_case(case, model_pair)


def evaluate_unsanitary_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Unsanitary landfill vs sanitary landfill for MSW."""
    case = EvaluationCase(
        category="unsanitary_gap",
        description="Unsanitary landfill of municipal solid waste vs sanitary landfill.",
        query="unsanitary landfill municipal solid waste switzerland",
        positive_doc=Path("data/markdown/0a101fb1-33a4-37ea-a0c4-24af4acd8a8d.md"),
        contrast_doc=Path("data/markdown/3c723bbd-166e-370f-9f23-13a8f3234923.md"),
        positive_label="Unsanitary landfill (MSW, CH)",
        contrast_label="Sanitary landfill for waste wood (CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_residue_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Different landfill compartments for MSWI residues."""
    case = EvaluationCase(
        category="residue_gap",
        description="Residual material landfill vs slag compartment for MSWI residues.",
        query="mswi residues residual material landfill switzerland",
        positive_doc=Path("data/markdown/1e8eff10-7282-363e-afb6-b3df1118dd07.md"),
        contrast_doc=Path("data/markdown/d5654ebc-6f2e-3699-a0fe-e980710a0f7e.md"),
        positive_label="Residual material landfill (Type C)",
        contrast_label="Slag compartment landfill (Type D)",
    )
    return evaluate_case(case, model_pair)


def evaluate_feed_processing_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Processed feed vs primary crop production."""
    case = EvaluationCase(
        category="feed_processing_gap",
        description="Wheat feed processing vs wheat grain cultivation.",
        query="wheat feed production milling process",
        positive_doc=Path("data/markdown/7d37724f-d9f3-3c3d-b4e4-66c831df81d5.md"),
        contrast_doc=Path("data/markdown/3f8209d5-309f-3ab6-9014-c7051aad8cc2.md"),
        positive_label="Wheat feed production (RoW)",
        contrast_label="Wheat grain cultivation (RoW)",
    )
    return evaluate_case(case, model_pair)


def evaluate_market_location_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Different country electricity markets with the same product definition."""
    case = EvaluationCase(
        category="market_location_gap",
        description="Medium-voltage electricity market in Peru vs Portugal.",
        query="medium voltage electricity market peru 2020",
        positive_doc=Path("data/markdown/44e46a76-ab54-30f6-b108-ade33c015ff7.md"),
        contrast_doc=Path("data/markdown/19ec7b9c-1434-3da9-9a91-1f667d85f8e1.md"),
        positive_label="Market electricity, MV (Location: PE)",
        contrast_label="Market electricity, MV (Location: PT)",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_code_gap_at(model_pair: dict[str, str]) -> dict[str, object]:
    """Use country code 'AT' to disambiguate nearly identical market descriptions."""
    case = EvaluationCase(
        category="location_code_gap",
        description="Query with Ecoinvent code 'AT' should pick the Austrian MV market over India's western grid.",
        query="AT medium voltage electricity market",
        positive_doc=Path("data/markdown/2086d24c-da73-3c7c-89cc-0406762155fd.md"),
        contrast_doc=Path("data/markdown/08faf4be-a6d7-3110-a6cd-de082166dc21.md"),
        positive_label="Market electricity, MV (Location code: AT)",
        contrast_label="Market electricity, MV (Location: IN-Western grid)",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_code_gap_pt(model_pair: dict[str, str]) -> dict[str, object]:
    """Country code query 'PT' should match Portugal instead of Peru."""
    case = EvaluationCase(
        category="location_code_gap",
        description="Stress-test two nearly identical MV markets using the ISO code PT vs Peru dataset.",
        query="PT medium voltage electricity market",
        positive_doc=Path("data/markdown/19ec7b9c-1434-3da9-9a91-1f667d85f8e1.md"),
        contrast_doc=Path("data/markdown/44e46a76-ab54-30f6-b108-ade33c015ff7.md"),
        positive_label="Market electricity, MV (Location code: PT)",
        contrast_label="Market electricity, MV (Location: PE)",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_code_gap_in_east(model_pair: dict[str, str]) -> dict[str, object]:
    """Regional grid code 'IN-Eastern grid' vs foreign grid."""
    case = EvaluationCase(
        category="location_code_gap",
        description="Differentiate India's eastern grid code from China's East Coast grid market.",
        query="IN-Eastern grid medium voltage electricity market",
        positive_doc=Path("data/markdown/7f8d9b5a-4c0e-3700-811a-d95102796376.md"),
        contrast_doc=Path("data/markdown/2b1c02f1-0990-3be2-8a72-895dd70d333c.md"),
        positive_label="Market electricity, MV (Location code: IN-Eastern grid)",
        contrast_label="Market electricity, MV (Location: CN-ECGC)",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_code_gap_is(model_pair: dict[str, str]) -> dict[str, object]:
    """Short ISO code 'IS' vs continental ES market."""
    case = EvaluationCase(
        category="location_code_gap",
        description="Icelandic MV market (code IS) should outrank the Spanish market when code is used in the query.",
        query="IS medium voltage electricity market",
        positive_doc=Path("data/markdown/03ce6261-0408-3f6f-8108-111c38584067.md"),
        contrast_doc=Path("data/markdown/5bcc1401-d29b-3fbb-ad40-1fa9435da19d.md"),
        positive_label="Market electricity, MV (Location code: IS)",
        contrast_label="Market electricity, MV (Location: ES)",
    )
    return evaluate_case(case, model_pair)


def evaluate_location_code_gap_in_south(model_pair: dict[str, str]) -> dict[str, object]:
    """Regional grid 'IN-Southern grid' vs Canadian province code 'CA-NS'."""
    case = EvaluationCase(
        category="location_code_gap",
        description="Ensure the query using the Southern India grid code is not hijacked by Nova Scotia.",
        query="IN-Southern grid medium voltage electricity market",
        positive_doc=Path("data/markdown/f2e71119-b26a-3345-8a7c-de5999024eb8.md"),
        contrast_doc=Path("data/markdown/669cbea7-1bc1-3d53-b8c4-05f558e9d53e.md"),
        positive_label="Market electricity, MV (Location code: IN-Southern grid)",
        contrast_label="Market electricity, MV (Location: CA-NS)",
    )
    return evaluate_case(case, model_pair)


# Hard-negative style checks where baseline margins flip sign pre/post fine-tuning.
def evaluate_ccgt_province_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Differentiate two combined-cycle gas plants that only vary by Chinese province."""
    case = EvaluationCase(
        category="ccgt_province_gap",
        description="Guangxi CCGT emissions vs highly similar Fujian plant.",
        query="electricity generation emission factors combined cycle gas turbine China",
        positive_doc=Path("data/markdown/6f196113-af10-3e21-bad8-2bece4119a4b.md"),
        contrast_doc=Path("data/markdown/cd285cc4-8033-39dd-9b7d-e3ba3ebd9450.md"),
        positive_label="Combined-cycle plant (Location: CN-GX)",
        contrast_label="Combined-cycle plant (Location: CN-FJ)",
    )
    return evaluate_case(case, model_pair)


def evaluate_sludge_chain_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Market mix for tin-sheet sludge vs Latvia landfarming of MSW leachate sludge."""
    case = EvaluationCase(
        category="sludge_chain_gap",
        description="Global WWT market for 97% water sludge vs LV landfarming disposal.",
        query="sewage sludge treatment LCA data 97% water",
        positive_doc=Path("data/markdown/558102aa-45b5-3b05-a8d8-a87cb2afe51d.md"),
        contrast_doc=Path("data/markdown/93128adc-20c9-3a12-a0f7-8107f8428668.md"),
        positive_label="WWT market mix (97% water sludge, GLO)",
        contrast_label="Landfarming chain (Location: LV)",
    )
    return evaluate_case(case, model_pair)


def evaluate_pellet_equipment_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Equipment manufacturing vs heat production activity for pellet boilers."""
    case = EvaluationCase(
        category="pellet_equipment_gap",
        description="Central Europe pellet boiler production vs Swiss furnace operation.",
        query="pellet boiler production emissions CO2 Central Europe",
        positive_doc=Path("data/markdown/4a13b02f-8889-3ce6-b081-fd4e93d86745.md"),
        contrast_doc=Path("data/markdown/a7cd58b0-d2a2-32d0-bbd7-0543a67fc85a.md"),
        positive_label="Pellet furnace manufacturing (300kW, CH)",
        contrast_label="Pellet heat production (25kW, CH)",
    )
    return evaluate_case(case, model_pair)


def evaluate_landfill_specificity_gap(model_pair: dict[str, str]) -> dict[str, object]:
    """Waste-specific lignite ash landfill vs Swiss leachate market mix."""
    case = EvaluationCase(
        category="landfill_specific_gap",
        description="Composition-specific landfill for lignite ash vs CH leachate market.",
        query="LCA dataset waste composition-specific landfill leachate and gas emissions",
        positive_doc=Path("data/markdown/64b4901d-e807-337f-aa8f-cfff7e2a18f3.md"),
        contrast_doc=Path("data/markdown/97ab922a-ebf0-36b3-87b2-77ad172ba78a.md"),
        positive_label="Lignite ash sanitary landfill (Composition specific)",
        contrast_label="Leachate market mix (Location: CH)",
    )
    return evaluate_case(case, model_pair)


CATEGORY_EVALUATORS: list[Callable[[dict[str, str]], dict[str, object]]] = [
    evaluate_region_difference,
    # evaluate_data_type_difference,
    evaluate_time_difference,
    evaluate_technology_difference,
    evaluate_location_difference,
    evaluate_disposal_method_difference,
    evaluate_market_vs_process,
    evaluate_attribute_mix_gap,
    evaluate_hazard_vs_nonhazard,
    evaluate_supply_chain_stage,
    evaluate_energy_source_gap,
    evaluate_circularity_gap,
    evaluate_aluminium_chain_gap,
    evaluate_waste_market_gap,
    evaluate_infrastructure_gap,
    evaluate_maintenance_gap,
    evaluate_clinker_market_gap,
    evaluate_unsanitary_gap,
    evaluate_residue_gap,
    evaluate_feed_processing_gap,
    evaluate_market_location_gap,
    evaluate_location_code_gap_at,
    evaluate_location_code_gap_pt,
    evaluate_location_code_gap_in_east,
    evaluate_location_code_gap_is,
    evaluate_location_code_gap_in_south,
    evaluate_ccgt_province_gap,
    evaluate_sludge_chain_gap,
    evaluate_pellet_equipment_gap,
    evaluate_landfill_specificity_gap,
]


def summarize_results(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for res in results:
        name = res["model_pair"]
        entry = summary.setdefault(
            name,
            {
                "count": 0,
                "total_margin_gain": 0.0,
                "total_positive_gain": 0.0,
                "total_contrast_gain": 0.0,
                "improved_cases": 0,
            },
        )
        entry["count"] += 1
        entry["total_margin_gain"] += float(res["margin_gain"])
        entry["total_positive_gain"] += float(res["positive_gain"])
        entry["total_contrast_gain"] += float(res["contrast_gain"])
        if res["margin_gain"] > 0:
            entry["improved_cases"] += 1

    for entry in summary.values():
        count = entry["count"] or 1
        entry["avg_margin_gain"] = entry["total_margin_gain"] / count
        entry["avg_positive_gain"] = entry["total_positive_gain"] / count
        entry["avg_contrast_gain"] = entry["total_contrast_gain"] / count
        entry["improvement_rate"] = entry["improved_cases"] / count
    return summary


def print_case_result(res: dict[str, object]) -> None:
    print(f"\n[{res['model_pair']}] {res['category']}: {res['description']}")
    print(f"  + Query: {res['query']}")
    print(
        f"  + Baseline -> pos={res['baseline']['positive']:.4f} "
        f"neg={res['baseline']['contrast']:.4f} margin={res['baseline']['margin']:.4f}"
    )
    print(
        f"  + Finetuned -> pos={res['finetuned']['positive']:.4f} "
        f"neg={res['finetuned']['contrast']:.4f} margin={res['finetuned']['margin']:.4f}"
    )
    print(
        f"    Δmargin={res['margin_gain']:.4f} "
        f"(Δpos={res['positive_gain']:.4f}, Δneg={res['contrast_gain']:.4f})"
    )


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print("Aggregate margin gains")
    print("=" * 72)
    header = f"{'Model':<10} {'Avg Δmargin':>12} {'Improvement':>14} {'Cases':>6}"
    print(header)
    print("-" * len(header))
    for model, stats in summary.items():
        print(
            f"{model:<10} "
            f"{stats['avg_margin_gain']:>12.4f} "
            f"{stats['improvement_rate']*100:>13.1f}% "
            f"{int(stats['count']):>6}"
        )


def print_metric_notes() -> None:
    print("\n" + METRIC_EXPLANATION_ZH.strip())


def write_embedded_html(output_path: Path, payload: dict[str, object]) -> None:
    if not VIEWER_TEMPLATE.exists():
        raise FileNotFoundError(f"Viewer template not found: {VIEWER_TEMPLATE}")
    template = VIEWER_TEMPLATE.read_text(encoding="utf-8")
    if EMBED_PLACEHOLDER not in template:
        raise ValueError("Template missing embedded data placeholder.")
    embedded_json = json.dumps(payload, ensure_ascii=False, indent=2)
    embedded_json = embedded_json.replace("</script>", "<\\/script>")
    replacement = (
        '<script id="embedded-data" type="application/json">\n'
        f"{embedded_json}\n"
        "</script>"
    )
    content = template.replace(EMBED_PLACEHOLDER, replacement, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Embedded HTML saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs fine-tuned embedding models on contrastive LCA scenarios."
    )
    parser.add_argument(
        "--model_pairs",
        type=str,
        default="qwen",
        help=f"Comma-separated list of model pair keys (available: {', '.join(MODEL_PAIRS)}).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/evaluate_similarity_gaps.json",
        help="Optional path to save detailed JSON results.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="docs/similarity_gap_report.html",
        help="Optional path to save an HTML viewer with embedded data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_keys = [key.strip() for key in args.model_pairs.split(",") if key.strip()]
    if not selected_keys:
        raise ValueError("No model pairs selected.")

    results: list[dict[str, object]] = []
    for key in selected_keys:
        if key not in MODEL_PAIRS:
            raise KeyError(f"Unknown model pair key '{key}'. Available: {', '.join(MODEL_PAIRS)}")
        pair = MODEL_PAIRS[key]
        print("\n" + "=" * 72)
        print(f"Evaluating model pair: {pair['name']} (baseline vs finetuned)")
        print("=" * 72)
        for evaluator in CATEGORY_EVALUATORS:
            res = evaluator(pair)
            results.append(res)
            print_case_result(res)

    summary = summarize_results(results)
    print_summary(summary)
    print_metric_notes()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cases": results,
            "summary": summary,
            "model_pairs": {key: MODEL_PAIRS[key] for key in selected_keys},
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDetailed report saved to {output_path}")
    else:
        payload = {
            "cases": results,
            "summary": summary,
            "model_pairs": {key: MODEL_PAIRS[key] for key in selected_keys},
        }

    if args.output_html:
        html_path = Path(args.output_html)
        write_embedded_html(html_path, payload)


if __name__ == "__main__":
    main()
