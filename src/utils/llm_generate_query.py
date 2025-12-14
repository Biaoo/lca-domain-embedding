from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.resolve()
import sys

sys.path.append(str(root_dir))

from openai import OpenAI
from src.config.model import DASHSCOPE_API_KEY, DASHSCOPE_API_URL


client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_URL,
)


SYSTEM_PROMPT_PROCESS = """You are a search query generation expert in the LCA (Life Cycle Assessment) domain. Your task is to generate potential search queries that users might use to find the given LCA process dataset.

Generate 5 diverse search queries. Consider these perspectives (but don't limit to one per query):
- **Material/Process**: material name, product category, process type (e.g., "steel production", "PET bottle recycling")
- **Geographic/Regional**: location or region (e.g., "electricity generation in Germany", "cement production Europe")
- **Technical/Environmental**: environmental impact, emission type, technical parameters (e.g., "CO2 emissions from incineration", "NOx emission factor")
- **Application scenario**: use case or industry (e.g., "LCA data for packaging industry", "carbon footprint construction")
- **Complex multi-criteria**: combine multiple factors (e.g., "waste incineration emission factors Europe", "aluminum production environmental impact")

Guidelines:
- Leverage any location, technology, reference flow/amount, inputs/outputs, and time coverage to keep queries grounded in the dataset.
- Mix simple keywords and complex queries naturally; include both technical terminology and common expressions.
- Simulate real user search behavior with varying expertise levels.
- Make queries realistic and specific to the dataset content.
- Do NOT include numeric amounts or reference quantities (e.g., masses/flows) in the queries.

Output strictly in the following JSON format without any additional content:
{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4",
        "query 5"
    ]
}"""


SYSTEM_PROMPT_FLOW = """You are a search query generation expert in the LCA (Life Cycle Assessment) domain. Your task is to generate potential search queries that users might use to find the given LCA flow dataset.

Generate 5 diverse search queries. Consider these perspectives (but don't limit to one per query):
- **Chemical/Material identity**: exact chemical names, synonyms, CAS/EC numbers, molecular descriptors.
- **Medium & classification**: emission to air/soil/water, resource input, elementary flow category; include the medium when relevant.
- **Usage/sector context**: typical applications or industries where the substance appears (e.g., pesticides, solvents, metallurgy, pharmaceuticals).
- **Environmental/health aspects**: toxicity, hazard, regulation, emission factors.
- **Complex combos**: combine name/synonym + CAS + medium/sector (e.g., "4433-79-8 emission to soil", "organic solvent CAS search water discharge").

Guidelines:
- Prefer identifiers present in the dataset (name, synonyms, CAS, EC) and avoid inventing properties not provided.
- Mix short and long queries; include both technical and common phrasing.
- Keep queries specific to this flow and likely user intents to retrieve it.
- Do NOT include numeric amounts/reference quantities (e.g., property mean) in the queries.

Output strictly in the following JSON format without any additional content:
{
    "queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4",
        "query 5"
    ]
}"""


def build_system_prompt(dataset_type: str | None) -> str:
    if (dataset_type or "").lower() == "flow":
        return SYSTEM_PROMPT_FLOW
    return SYSTEM_PROMPT_PROCESS


def generate_query(dataset_content: str, dataset_type: str | None = None):
    system_prompt = build_system_prompt(dataset_type)

    user_prompt = f"""Based on the following LCA dataset description, generate 5 search queries that users might use to find this dataset:

---
{dataset_content}
---

Please output 5 search queries in JSON format."""

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "search_queries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 5,
                            "maxItems": 5,
                            "description": "5 search queries that users might use to find this LCA dataset"
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False
                }
            }
        },
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    queries = generate_query("""# treatment of municipal solid waste, open burning | municipal solid waste | Cutoff, U

**UUID:** `0a0bf34e-6af3-3f63-a1f5-6032e1eca2e5`

**Classification:** 3821:Treatment and disposal of non-hazardous waste

## Description

This dataset represents the activity of waste disposal of 'municipal solid waste' in an uncontrolled, open burning. Recommended use of this dataset: open burning of mixed municipal waste.;The inventoried waste contains several fractions: 13.7% Textiles, leather, rubber; 0.289% newsprint; 20.1% sanitary products; 1.06% wood; 3.17% vacuum cleaner bags; 6.02% animal-derived food waste; 20.1% vegetable food waste; 0.00329% glass packaging-clear; 0.000548% glass packaging-green; 4.78% glass packaging...

## Time Coverage

Reference Year: 2006 | Valid Until: 2024

## Geography

**Location:** PT
Modelling parameters representing a generic open burning process, not geographically specific.

## Technology

This is an inventory for open burning of solid waste, representing an informal and uncontrolled burning. No flue gas treatment. Suitable for informal recycling or disposal. Based on Municipal waste incineration without any pollution control. Residues (minus any actually recycled metals) are left on site. Assumption of share of thermal NOx to total NOx in average waste burning process is 30%. Contribution from thermal NOx is 810mg per kg waste. Recycling rates for metallic iron = 0% for metallic aluminium = 0% for metallic copper = 0%.

## Methodology

**Data Set Type:** Unit process, single operation
**LCI Method:** Other

## Outputs

- Carbon monoxide, non-fossil: 0.0251
- Nitrogen oxides: 0.01111
- Non-hazardous waste disposed: 1
- Aluminium III: 0.01283
- Carbon dioxide, non-fossil: 0.6522
- Silicon: 0.06801
- Heat, waste: 19.93
- Calcium II: 0.02226
- Sodium I: 0.01457
- Carbon dioxide, fossil: 0.3509

## Main Inputs

- Waste mass, total, placed in landfill: 0.2933
- Oxygen: 0.9046

**Version:** 03.11.000""")
    
    print(queries)
