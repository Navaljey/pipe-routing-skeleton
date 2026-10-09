"""검증기 — 고정 심판 (§6, D12, D44). 라우터와 독립. 사용: verify(scenario, [PipeRoute, ...])."""
from .core import MODULE_ORDER, MODULES, PipeReport, PipeRoute, Violation, verify  # noqa: F401
from . import modules  # noqa: F401  (Layer 0 모듈 등록)
