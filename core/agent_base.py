"""
Agent 基类 — 为所有 Agent 提供通信能力

每个 Agent 获得：
- self.bus          信号总线引用
- self.emit()       发送信号
- self.listen()     订阅信号
- self.agent_name   身份标识
"""

from core.signal_bus import SignalBus, Signal


class BaseAgent:
    """所有 Agent 的基类"""

    agent_name: str = "UnnamedAgent"

    def __init__(self):
        self.bus = SignalBus()
        self._setup_listeners()

    def _setup_listeners(self):
        """子类覆盖此方法来注册信号监听器"""
        pass

    def emit(self, signal_name: str, payload: dict = None) -> None:
        signal = Signal(
            name=signal_name,
            sender=self.agent_name,
            payload=payload or {},
        )
        self.bus.publish(signal)

    def listen(self, signal_name: str, handler) -> None:
        self.bus.subscribe(signal_name, handler)
