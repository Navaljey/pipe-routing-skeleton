"""D101 Σ 단독 J 참고치 — 재사용이 결과 불변인지, 출력 필드, member_blockage (R7) 필드."""
import pytest

from pipe_routing.layered import make_layered_router
from pipe_routing.pipeline import run
from pipe_routing.scenario import load
from pipe_routing.solo_ref import attach, solo_routes
from pipe_routing.support_std import load_standard


@pytest.fixture(scope="module")
def manual():
    sc = load("scenarios/manual/manual_01.json")
    router = make_layered_router()
    return sc, router, run(sc, router=router)


def test_fields(manual):
    sc, _, out = manual
    s = out["global"]["solo_reference"]
    assert s["definition"] == "D101" and s["both_pass"] >= 1
    assert s["solo_fail"] == sum(1 for x in out["routes"] if not x["solo"]["pass"])
    assert all("solo" in x for x in out["routes"])
    mb = out["global"]["member_blockage"]
    assert set(mb) >= {"count", "support_violation", "after", "before", "pipes"}


def test_reuse_is_result_invariant(manual):
    sc, router, out = manual
    assert out["global"]["solo_reference"]["reused"] >= 1      # P005: 분류 단계의 예약 없는 단독 결과
    fresh, reused = solo_routes(sc, router, reuse=None)
    assert reused == 0
    for x in out["routes"]:
        r = fresh[x["pipe_id"]]
        assert r.status == x["solo"]["status"] and int(r.expanded) == x["solo"]["expanded"]


def test_deterministic(manual):
    sc, router, out = manual
    out2 = {"routes": [dict(x) for x in out["routes"]], "global": dict(out["global"])}
    attach(out2, sc, router, load_standard())
    a = {k: v for k, v in out["global"]["solo_reference"].items() if k not in ("sec", "reused")}
    b = {k: v for k, v in out2["global"]["solo_reference"].items() if k not in ("sec", "reused")}
    assert a == b
    assert [x["solo"] for x in out["routes"]] == [x["solo"] for x in out2["routes"]]
