"""
因子注册表模块（适配自 Ray 的 factor_framework/registry.py）

核心功能：
- 单例注册表，管理所有因子类和元信息
- @register_factor 装饰器，一行注册
- 支持按类别/状态/作者过滤
- JSON 持久化元信息
"""

import copy
import json
from pathlib import Path
from typing import Optional, List

import pandas as pd


class FactorRegistry:
    """因子注册表单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}      # {name: factor_class}
            cls._instance._metadata = {}     # {name: FactorMetadata}
        return cls._instance

    def register(self, factor_class, metadata) -> None:
        name = metadata.name
        self._factors[name] = factor_class
        self._metadata[name] = metadata

    def get_class(self, name: str):
        if name not in self._factors:
            raise KeyError(f"因子 {name} 未注册")
        return self._factors[name]

    def get_metadata(self, name: str):
        if name not in self._metadata:
            raise KeyError(f"因子 {name} 未注册")
        return self._metadata[name]

    def create_instance(self, name: str):
        """创建因子实例（深拷贝 metadata 防止共享污染）"""
        cls = self.get_class(name)
        meta = self.get_metadata(name)
        return cls(copy.deepcopy(meta))

    def update_metadata(self, name: str, metadata) -> None:
        if name not in self._metadata:
            raise KeyError(f"因子 {name} 未注册")
        self._metadata[name] = metadata

    def list_factors(
        self,
        category: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[str]:
        """列出因子名称，支持过滤"""
        results = []
        for name, meta in self._metadata.items():
            if category and meta.category != category:
                continue
            if enabled_only and not meta.enabled:
                continue
            results.append(name)
        return results

    def get_enabled_factors(self) -> dict:
        """返回所有启用因子的 {name: metadata}"""
        return {
            name: meta for name, meta in self._metadata.items()
            if meta.enabled
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self._metadata:
            return pd.DataFrame()
        rows = [meta.to_dict() for meta in self._metadata.values()]
        return pd.DataFrame(rows)

    def save_to_disk(self, path: str) -> None:
        """保存注册表到 JSON"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {name: meta.to_dict() for name, meta in self._metadata.items()}
        # 原子写入
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    def load_from_disk(self, path: str) -> None:
        from factor_framework.base import FactorMetadata
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, meta_dict in data.items():
            if name not in self._metadata:
                self._metadata[name] = FactorMetadata.from_dict(meta_dict)

    def summary(self) -> str:
        total = len(self._metadata)
        if total == 0:
            return "因子库为空"
        categories = {}
        for meta in self._metadata.values():
            categories[meta.category] = categories.get(meta.category, 0) + 1
        lines = [f"因子库概览: 共 {total} 个因子"]
        for cat, cnt in sorted(categories.items()):
            lines.append(f"  {cat}: {cnt}")
        enabled = sum(1 for m in self._metadata.values() if m.enabled)
        lines.append(f"  启用: {enabled}/{total}")
        return "\n".join(lines)


def register_factor(metadata):
    """装饰器：注册因子类到全局注册表"""
    def decorator(cls):
        FactorRegistry().register(cls, metadata)
        return cls
    return decorator
