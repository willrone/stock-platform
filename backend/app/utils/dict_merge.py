"""
深度合并工具
"""


def deep_merge(base: dict, override: dict) -> dict:
    """
    递归深度合并两个字典，override 覆盖 base。
    
    规则：
    - 两个值都是 dict → 递归合并
    - 类型不同 → override 覆盖
    - override 有但 base 没有 → 新增
    - 列表类型（如 stock_codes）→ 直接替换，不 append
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
