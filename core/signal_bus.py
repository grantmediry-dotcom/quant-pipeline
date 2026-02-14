"""
信号总线 — 轻量级 Agent 间通信

设计原则：
- 同步发布/订阅（无线程，无异步）
- 所有信号经过总线，可审计可追溯
- Agent 不直接引用彼此，只通过信号名解耦

信号命名约定："{发送方}.{事件}"  例如 "monitor.factor_degraded"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from collections import defaultdict, Counter


@dataclass
class Signal:
    """一条信号消息"""
    name: str           # 信号名，如 "monitor.factor_degraded"
    sender: str         # 发送者名称，如 "MonitorAgent"
    payload: dict       # 数据载荷
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SignalBus:
    """
    全局信号总线（单例）

    用法：
        bus = SignalBus()
        bus.subscribe("monitor.factor_degraded", my_handler)
        bus.publish(Signal(name="monitor.factor_degraded", sender="Monitor", payload={...}))
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = defaultdict(list)
            cls._instance._history = []
        return cls._instance

    def subscribe(self, signal_name: str, callback: Callable[[Signal], None]) -> None:
        self._subscribers[signal_name].append(callback)

    def publish(self, signal: Signal) -> None:
        self._history.append(signal)
        print(f"  [SignalBus] {signal.sender} -> {signal.name}")

        for callback in self._subscribers.get(signal.name, []):
            try:
                callback(signal)
            except Exception as e:
                print(f"  [SignalBus] 处理信号异常: {signal.name} -> {e}")

    def get_history(self) -> list:
        return list(self._history)

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()

    def summary(self) -> str:
        lines = [f"信号总线: 共 {len(self._history)} 条信号"]
        counts = Counter(s.name for s in self._history)
        for name, cnt in counts.most_common():
            lines.append(f"  {name}: {cnt}")
        return "\n".join(lines)
