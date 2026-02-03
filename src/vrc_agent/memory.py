from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    timestamp: str
    scene: str
    heard: str
    speak: str
    actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "scene": self.scene,
            "heard": self.heard,
            "speak": self.speak,
            "actions": self.actions,
        }


class MemoryStore:
    def __init__(self, file_path: str, max_records: int = 1000):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max(10, int(max_records))

    def append(self, item: MemoryItem) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        self._truncate_if_needed()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        rows = self._load_all()
        if not rows:
            return []
        k = max(1, int(top_k))
        q_tokens = self._tokenize(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        total = len(rows)
        for idx, row in enumerate(rows):
            text = f"{row.get('scene', '')}\n{row.get('heard', '')}\n{row.get('speak', '')}"
            overlap = self._overlap_score(q_tokens, self._tokenize(text))
            recency = (idx + 1) / total
            score = overlap * 0.85 + recency * 0.15
            scored.append((score, idx, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[2] for x in scored[:k] if x[0] > 0.0]

    def _load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    def _truncate_if_needed(self) -> None:
        rows = self._load_all()
        if len(rows) <= self.max_records:
            return
        keep = rows[-self.max_records :]
        with self.path.open("w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = text.lower()
        # Include both latin words and CJK chunks for simple cross-language matching.
        parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,3}", text)
        return {p for p in parts if p}

    @staticmethod
    def _overlap_score(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / max(1, len(a))


def build_memory_item(scene: str, heard: str, speak: str, actions: list[dict[str, Any]]) -> MemoryItem:
    return MemoryItem(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        scene=scene,
        heard=heard,
        speak=speak,
        actions=actions,
    )
