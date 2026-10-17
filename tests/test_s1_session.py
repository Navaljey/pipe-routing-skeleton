"""S1-6b 증분 검증 (D85) — 무작위 추가·제거·변경 순서에서 매 단계 전체 재검증과 출력 비트 동일."""
import json
import random
import unittest

from pipe_routing.verifier import PipeRoute, Session, verify
from tests.test_s1_support import SYN
from tests.test_verifier import pipe, scenario

SIZES = ["25A", "65A", "100A", "150A", "200A"]


def dump(reports) -> str:
    return json.dumps({k: v.to_dict() for k, v in reports.items()}, sort_keys=True, default=str)


def make(seed: int, n: int = 6):
    """x=0 → x=40000 경계 배관 n 개 — 좁은 띠(y 18~22 m, z 0.8~1.6 m)에 몰려 서로 간섭·D 묶음·부재 막힘이 생긴다."""
    rng = random.Random(seed)
    pipes, routes = [], {}
    for i in range(n):
        y0 = rng.randrange(18000, 22000, 10)
        z = rng.randrange(800, 1600, 10)
        pipes.append(pipe(f"P{i:02d}", size=rng.choice(SIZES), start=[0, y0, z], end=[40000, y0, z]))
        routes[f"P{i:02d}"] = (y0, z)
    obstacles = [([15000, 19500, 0], [16000, 20500, 600]), ([25000, 18000, 2500], [27000, 22000, 3000])]
    return scenario(pipes, obstacles), routes, rng


def random_route(rng, y0, z):
    """Z 모양: 직진 → y 로 옆걸음 → 다시 y0 로 돌아와 끝 단자 (단자 좌표·방향은 유지)."""
    if rng.random() < 0.3:
        return [[0, y0, z], [40000, y0, z]]
    x1 = rng.randrange(5000, 15000, 10)
    x2 = rng.randrange(25000, 35000, 10)
    y1 = y0 + rng.choice([-1, 1]) * rng.randrange(300, 1500, 10)
    return [[0, y0, z], [x1, y0, z], [x1, y1, z], [x2, y1, z], [x2, y0, z], [40000, y0, z]]


class SessionTest(unittest.TestCase):
    def check(self, sc, sess, placed):
        ref = verify(sc, [PipeRoute(p.id, placed[p.id]) for p in sc.pipes if p.id in placed], std=SYN)["pipes"]
        self.assertEqual(dump(sess.reports()), dump(ref))
        res = sess.result()
        jt = sum(v["J"]["total"] for v in res["pipes"].values() if v["success"])
        self.assertAlmostEqual(res["J_total"], round(jt, 4), places=4)

    def test_random_sequences_bit_identical(self):
        grouped = blocked = 0
        for seed in range(2):
            sc, ends, rng = make(seed)
            sess = Session(sc, std=SYN)
            placed = {}
            for step in range(15):
                free = [p for p in ends if p not in placed]
                op = rng.choice(["add"] * 3 + ["replace"] * 2 + ["remove"]) if placed else "add"
                if op == "add" and free:
                    pid = rng.choice(free)
                    placed[pid] = random_route(rng, *ends[pid])
                    sess.add(pid, placed[pid])
                elif op == "replace" or (op == "add" and not free):
                    pid = rng.choice(sorted(placed))
                    placed[pid] = random_route(rng, *ends[pid])
                    sess.replace(pid, placed[pid])
                else:
                    pid = rng.choice(sorted(placed))
                    del placed[pid]
                    sess.remove(pid)
                self.check(sc, sess, placed)
                reps = sess.reports()
                grouped += sum(1 for r in reps.values() for s in r.supports if s.get("form") == "D")
                blocked += sum(1 for r in reps.values() if r.layer0.get("support") is False)
        self.assertGreater(grouped, 0)      # D 묶음 경로를 실제로 탔는지
        self.assertGreater(blocked, 0)      # 다른 배관에 의한 서포트 막힘도

    def test_far_pipe_not_recomputed(self):
        """영향 범위 밖 배관은 다시 판정하지 않고, 가까운 배관은 영향받는 모듈만 다시 판정한다 (D87 ⑤)."""
        a = pipe("A", start=[0, 5000, 1000], end=[40000, 5000, 1000])
        b = pipe("B", start=[0, 35000, 1000], end=[40000, 35000, 1000])
        c = pipe("C", start=[0, 5600, 1000], end=[40000, 5600, 1000])     # A 서포트 부재 영역 밖 (가로 600)
        d = pipe("D", start=[0, 5300, 1000], end=[40000, 5300, 1000])     # A 다리 자리 안 (가로 300)
        sc = scenario([a, b, c, d])
        sess = Session(sc, std=SYN)
        self.assertEqual(sess.add("A", [[0, 5000, 1000], [40000, 5000, 1000]])["recomputed"], ["A"])
        self.assertEqual(sess.add("B", [[0, 35000, 1000], [40000, 35000, 1000]])["recomputed"], ["B"])
        self.assertEqual(sess.add("C", [[0, 5600, 1000], [40000, 5600, 1000]])["recomputed"], ["C"])
        r = sess.add("D", [[0, 5300, 1000], [40000, 5300, 1000]])
        self.assertEqual(r["recomputed_modules"]["A"], ["support"])
        self.assertIn("C", r["recomputed"])
        ref = verify(sc, [PipeRoute(p, w) for p, w in [("A", [[0, 5000, 1000], [40000, 5000, 1000]]),
                                                       ("B", [[0, 35000, 1000], [40000, 35000, 1000]]),
                                                       ("C", [[0, 5600, 1000], [40000, 5600, 1000]]),
                                                       ("D", [[0, 5300, 1000], [40000, 5300, 1000]])]], std=SYN)
        self.assertEqual(dump(sess.reports()), dump(ref["pipes"]))
        self.assertNotIn("B", sess.remove("D")["recomputed"])

    def test_errors(self):
        sc, ends, _ = make(1, 2)
        sess = Session(sc, std=SYN)
        with self.assertRaises(KeyError):
            sess.remove("P00")
        sess.add("P00", [[0, ends["P00"][0], ends["P00"][1]], [40000, ends["P00"][0], ends["P00"][1]]])
        with self.assertRaises(KeyError):
            sess.add("P00", [[0, ends["P00"][0], ends["P00"][1]], [40000, ends["P00"][0], ends["P00"][1]]])


if __name__ == "__main__":
    unittest.main()
