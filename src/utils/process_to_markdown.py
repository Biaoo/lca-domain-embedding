"""将 TidasProcess 对象转换为 Markdown 文本"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tidas_sdk import TidasProcess


def _get_text_from_dict(data: Any) -> Optional[str]:
    """从字典格式的多语言字段中提取文本

    Args:
        data: 字典格式的数据，如 {'@xml:lang': 'en', '#text': '...'}

    Returns:
        提取的文本值，如果为空则返回 None
    """
    if data is None:
        return None
    if isinstance(data, str):
        return data.strip() if data.strip() else None
    if isinstance(data, dict):
        text = data.get("#text", "")
        return text.strip() if text and text.strip() else None
    return None


def _get_flow_name(reference_to_flow: Dict) -> Optional[str]:
    """从流引用中提取流名称"""
    if not reference_to_flow:
        return None
    short_desc = reference_to_flow.get("common:shortDescription")
    return _get_text_from_dict(short_desc)


def _ensure_list(obj) -> list:
    """确保返回列表，如果是单个对象则包装为列表"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _get_reference_flow(process: "TidasProcess") -> Optional[Dict[str, Any]]:
    """根据 quantitativeReference 从 exchanges 中提取 reference flow

    Args:
        process: TidasProcess 对象

    Returns:
        包含 reference flow 信息的字典，如果找不到返回 None
    """
    pi = process.process_data_set.process_information
    qr = pi.quantitative_reference

    if not qr or qr.reference_to_reference_flow is None:
        return None

    ref_flow_id = qr.reference_to_reference_flow
    exchanges = process.process_data_set.exchanges

    if not exchanges or not exchanges.exchange:
        return None

    exchange_list = _ensure_list(exchanges.exchange)
    for ex in exchange_list:
        # reference_to_reference_flow 可能是字符串或整数
        if str(ex.data_set_internal_id) == str(ref_flow_id):
            return {
                "name": _get_flow_name(ex.reference_to_flow_data_set),
                "direction": ex.exchange_direction,
                "amount": ex.mean_amount,
                "uuid": ex.reference_to_flow_data_set.get("@refObjectId") if ex.reference_to_flow_data_set else None,
            }

    return None


def tidas_process_to_markdown(process: "TidasProcess", lang: str = "en") -> str:
    """将 TidasProcess 对象转换为 Markdown 格式

    使用 TidasProcess 对象的内置属性和方法，选取数据集的主要特征字段，跳过空字段。

    Args:
        process: TidasProcess 对象实例
        lang: 首选语言，默认为英文

    Returns:
        Markdown 格式的文本
    """
    sections = []

    pds = process.process_data_set
    pi = pds.process_information
    dsi = pi.data_set_information

    # 1. 数据集名称
    if dsi.name and dsi.name.base_name:
        base_name = _get_text_from_dict(dsi.name.base_name)
        if base_name:
            sections.append(f"# {base_name}")

    # 2. UUID
    if dsi.common_uuid:
        sections.append(f"**UUID:** `{dsi.common_uuid}`")

    # 3. Reference Flow (功能单位/参考流)
    ref_flow = _get_reference_flow(process)
    if ref_flow and ref_flow.get("name"):
        ref_parts = [f"**Reference Flow:** {ref_flow['name']}"]
        if ref_flow.get("amount") is not None:
            ref_parts.append(f"**Amount:** {ref_flow['amount']}")
        sections.append("\n".join(ref_parts))

    # 4. 分类信息
    if dsi.classification_information and dsi.classification_information.common_classification:
        class_data = dsi.classification_information.common_classification.common_class
        class_text = _get_text_from_dict(class_data)
        if class_text:
            sections.append(f"**Classification:** {class_text}")

    # 5. 一般描述 (使用 MultiLangList 的 get_text 方法)
    if dsi.common_general_comment:
        general_comment = dsi.common_general_comment.get_text(lang)
        if general_comment:
            # 截取前 500 字符，避免过长
            if len(general_comment) > 500:
                general_comment = general_comment[:500] + "..."
            sections.append(f"## Description\n\n{general_comment}")

    # 6. 时间信息
    time_info = pi.time
    if time_info:
        time_parts = []
        if time_info.common_reference_year:
            time_parts.append(f"Reference Year: {time_info.common_reference_year}")
        if time_info.common_data_set_valid_until:
            time_parts.append(f"Valid Until: {time_info.common_data_set_valid_until}")
        if time_parts:
            sections.append(f"## Time Coverage\n\n" + " | ".join(time_parts))

    # 7. 地理信息
    if pi.geography and pi.geography.location_of_operation_supply_or_production:
        loc = pi.geography.location_of_operation_supply_or_production
        geo_text = []
        if loc.location:
            geo_text.append(f"**Location:** {loc.location}")
        if loc.description_of_restrictions:
            desc = _get_text_from_dict(loc.description_of_restrictions)
            if desc:
                geo_text.append(f"\n{desc}")
        if geo_text:
            sections.append(f"## Geography\n\n" + "".join(geo_text))

    # 8. 技术描述 (technology 可能是 dict 类型)
    if pi.technology:
        tech = pi.technology
        if isinstance(tech, dict):
            tech_desc = _get_text_from_dict(tech.get("technologyDescriptionAndIncludedProcesses"))
        elif hasattr(tech, "technology_description_and_included_processes"):
            tech_desc_obj = tech.technology_description_and_included_processes
            if hasattr(tech_desc_obj, "get_text"):
                tech_desc = tech_desc_obj.get_text(lang)
            else:
                tech_desc = _get_text_from_dict(tech_desc_obj)
        else:
            tech_desc = None
        if tech_desc:
            sections.append(f"## Technology\n\n{tech_desc}")

    # 9. 方法论信息
    if pds.modelling_and_validation and pds.modelling_and_validation.lci_method_and_allocation:
        lci = pds.modelling_and_validation.lci_method_and_allocation
        method_parts = []
        if lci.type_of_data_set:
            method_parts.append(f"**Data Set Type:** {lci.type_of_data_set}")
        if lci.lci_method_principle:
            method_parts.append(f"**LCI Method:** {lci.lci_method_principle}")
        if method_parts:
            sections.append(f"## Methodology\n\n" + "\n".join(method_parts))

    # 10. 数据来源信息
    if pds.modelling_and_validation:
        data_sources = pds.modelling_and_validation.data_sources_treatment_and_representativeness
        if data_sources and isinstance(data_sources, dict):
            sampling = _get_text_from_dict(data_sources.get("samplingProcedure"))
            if sampling:
                sections.append(f"## Data Sources\n\n**Sampling:** {sampling}")

    # 11. 交换流 (输入/输出，排除 reference flow)
    if pds.exchanges and pds.exchanges.exchange:
        ref_flow_id = None
        if pi.quantitative_reference:
            ref_flow_id = str(pi.quantitative_reference.reference_to_reference_flow)

        inputs_dict = {}  # flow_name -> total_amount
        outputs_dict = {}  # flow_name -> total_amount
        exchange_list = _ensure_list(pds.exchanges.exchange)

        for ex in exchange_list:
            # 跳过 reference flow
            if str(ex.data_set_internal_id) == ref_flow_id:
                continue

            flow_name = _get_flow_name(ex.reference_to_flow_data_set)
            amount = ex.mean_amount

            if flow_name and amount is not None:
                try:
                    amount_val = float(amount)
                    if ex.exchange_direction == "Input":
                        inputs_dict[flow_name] = inputs_dict.get(flow_name, 0) + amount_val
                    elif ex.exchange_direction == "Output":
                        outputs_dict[flow_name] = outputs_dict.get(flow_name, 0) + amount_val
                except (ValueError, TypeError):
                    pass

        # 按数量降序排序，过滤小于0.001的，取前10个
        if inputs_dict:
            sorted_inputs = sorted(
                [(name, amt) for name, amt in inputs_dict.items() if amt >= 0.001],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            if sorted_inputs:
                sections.append(f"## Main Inputs\n\n" + "\n".join(
                    f"- {name}: {amt:.4g}" for name, amt in sorted_inputs
                ))

        if outputs_dict:
            sorted_outputs = sorted(
                [(name, amt) for name, amt in outputs_dict.items() if amt >= 0.001],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            if sorted_outputs:
                sections.append(f"## Main Outputs\n\n" + "\n".join(
                    f"- {name}: {amt:.4g}" for name, amt in sorted_outputs
                ))



    # 12. 版本信息
    if pds.administrative_information and pds.administrative_information.publication_and_ownership:
        pub = pds.administrative_information.publication_and_ownership
        if pub.common_data_set_version:
            sections.append(f"**Version:** {pub.common_data_set_version}")

    return "\n\n".join(sections)
