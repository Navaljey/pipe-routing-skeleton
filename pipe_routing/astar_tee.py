"""D112·D134 분기 1개 정확 풀이용 컴파일 A* — `astar_fast._search_generic` 을 그대로 옮기고 티 전이만 더한 것.

기존 커널 (`_search_generic`) 은 바꾸지 않는다 — 분기 없는 배관은 지금과 비트 동일 (D134 ⑥).
더한 것:
  - 상태에 티 비트 t (0 = 티 전, 1 = 티 후). 상태 같음·지배는 t 가 같을 때만. 목표 = 도착 ∧ t = 1.
  - 티 전이: 확장한 상태 (노드 n, 방향 d, t = 0) 에서 teec[n·18 + d] 가 유한하고 티 앞 직관 run ≥ tee_pre[bend] 이며
    들어온 엣지 끝이 꺾임 없음을 허용 (마스크 비트 0) 하면, 같은 노드·같은 방향 상태 (run 0, bend 등급 4 = "티",
    마스크·h·u 그대로 — 티는 주관의 서포트 고정점 아님, Q3 (b)) 를 g + teec 로 만든다. teec = 다리 비용 (다리 쪽 한 점
    전 탐색 결과, tree.py).
  - bend 등급 5개 (0/45/90/135/티). 지배의 "bend ≤" 를 ble[a, b] (a 의 직관 요구가 b 의 것 이하 — need·goal_need·tee_pre
    모두) 로 바꾼다. 등급 0~3 끼리는 a ≤ b 와 같다.
"""
import numpy as np

from .astar_fast import HAVE_NUMBA, _round6, _arc_prefilter, _arc_callback_generic

if HAVE_NUMBA:
    import numba
    from numba import njit

    @njit(cache=True)
    def _search_tee(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc):
        """이웃 표 그래프용 A* 본체 (_search 와 같은 구조, 노드 = 정수 id). 탐색 상태는 호출 사이에 유지된다 (배열 + ctr/fctr). 반환: 코드.

        T = 그래프 표, C = 상수, G = 엘보 호 거르기 표.
        H = 우선순위 큐 (f, 삽입 순번, g, 상태 id) 배열 — (f, 순번) 최소 힙. 순번이 유일하므로 꺼내는 순서는
            파이썬 heapq 와 같다
        S = 상태 배열 (노드, 방향, 직진 길이, 편향각 인덱스, g_best, 부모, 같은 (노드, 방향) 다음 상태, 끝 꺾임 허용 마스크)
        K = 확정 목록 (run, bend, g, 다음, 마스크) — (노드, 방향)별 연결 리스트, 머리는 closed_head
        D61: 중력관은 상태에 마지막 하향 이후 수평 누적 h (지배 = h ≤ 도 포함). 압력관은 h = 0 그대로
        D60: 마스크 = 들어온 엣지의 끝에서 허용되는 꺾임 등급(0/45/90/135 → 비트 0~3). 엣지 emask[n, d] 의 비트
             (시작 등급 i × 4 + 끝 등급 j) = 시작·끝을 그 엘보 접선만큼 깎았을 때 놓인 배관과 이격 통과 (엘보 호가
             꺾임점을 깎아 지나가는 실제 형상). 지배 = run ≥ · bend ≤ · 마스크 ⊇ · g ≤
        open_head[(노드, 방향)] = 그 (노드, 방향)의 첫 상태 id — 파이썬 g_best 딕셔너리의 키 (노드, 방향, run, bend) 조회와 같다
        ctr: [expanded, generated, tie, pending_id, heap_size, found_id, n_states, n_closed]
        fctr: [pending_g]
        """
        (step, elen, scost, emask, px, py, pz, hdist, clearance, pipe_margin, end_flat, dkind, gpar, spack, smaxd, bendu,
         hlow, teec, tee_pre, ble) = T
        gflag = gpar[0] > 0.5   # D61 중력관
        hflag = gpar[2] > 0.5   # D126 공식 휴리스틱 인자 (노드별 하한 표) — 없으면 아래 식 그대로
        lmax = gpar[1]
        (defl_idx, dir_vec, need_m, run_cap, goal_need_m, thr, ekg, kgpm, kgmm, e45, end_dir, check_every,
         goal_x, goal_y, goal_z) = C
        hf, ht, hg, hid = H
        s_node, s_dir, s_run, s_bend, s_g, s_par, s_next, s_mask, s_h, s_u, s_t = S
        c_run, c_bend, c_g, c_next, c_mask, c_h, c_u, c_t = K
        while True:
            # 배열 여유: 한 번 펼칠 때 상태·큐는 최대 18개, 확정 목록은 1개 늘어난다
            if ctr[4] + 18 > hf.shape[0] or ctr[6] + 18 > s_node.shape[0] or ctr[7] + 1 > c_run.shape[0]:
                return 4
            if ctr[3] >= 0:
                cur = ctr[3]
                g = fctr[0]
            else:
                if ctr[4] == 0:
                    return 0
                # pop (f, 순번) 최소
                g = hg[0]
                cur = hid[0]
                n = ctr[4] - 1
                ctr[4] = n
                if n > 0:
                    lf, lt, lg, li = hf[n], ht[n], hg[n], hid[n]
                    pos = 0
                    while True:
                        c = 2 * pos + 1
                        if c >= n:
                            break
                        if c + 1 < n and (hf[c + 1] < hf[c] or (hf[c + 1] == hf[c] and ht[c + 1] < ht[c])):
                            c += 1
                        if hf[c] < lf or (hf[c] == lf and ht[c] < lt):
                            hf[pos], ht[pos], hg[pos], hid[pos] = hf[c], ht[c], hg[c], hid[c]
                            pos = c
                        else:
                            break
                    hf[pos], ht[pos], hg[pos], hid[pos] = lf, lt, lg, li
                if g > s_g[cur] + 1e-9:
                    continue
                cflat = s_node[cur]
                cd = s_dir[cur]
                crun, cb, cm, chz, cu, ct = s_run[cur], s_bend[cur], s_mask[cur], s_h[cur], s_u[cur], s_t[cur]
                ckey = cflat * 18 + cd
                dom = False
                h = closed_head[ckey]
                while h >= 0:
                    if c_t[h] == ct and c_run[h] >= crun and ble[c_bend[h], cb] and (c_mask[h] & cm) == cm \
                            and c_h[h] <= chz and c_u[h] <= cu and c_g[h] <= g + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                m = ctr[7]
                c_run[m], c_bend[m], c_g[m], c_next[m], c_mask[m], c_h[m], c_u[m] = crun, cb, g, closed_head[ckey], cm, chz, cu
                c_t[m] = ct
                closed_head[ckey] = m
                ctr[7] = m + 1
                ctr[0] += 1
                if ct == 1 and cflat == end_flat and cd == end_dir and crun >= goal_need_m[cb] and (cm & 1):
                    ctr[5] = cur
                    return 1
                if ctr[0] % check_every == 0:
                    ctr[3] = cur
                    fctr[0] = g
                    return 3
            ctr[3] = -1
            cflat = s_node[cur]
            cd = s_dir[cur]
            crun, cb, cm, chz, cu, ct = s_run[cur], s_bend[cur], s_mask[cur], s_h[cur], s_u[cur], s_t[cur]
            # D112·D134 티 전이: 같은 노드·같은 방향, 티 앞 직관 ≥ tee_pre[cb], 들어온 엣지 끝이 꺾임 없음 허용 (마스크 비트 0)
            tc = teec[cflat * 18 + cd]
            if ct == 0 and tc < 1e300 and crun >= tee_pre[cb] and (cm & 1):
                ng = g + tc
                nkey = cflat * 18 + cd
                sid = open_head[nkey]
                while sid >= 0:
                    if s_t[sid] == 1 and s_run[sid] == 0.0 and s_bend[sid] == 4 and s_mask[sid] == cm \
                            and s_h[sid] == chz and s_u[sid] == cu:
                        break
                    sid = s_next[sid]
                if sid < 0 or ng + 1e-9 < s_g[sid]:
                    dom = False
                    h = closed_head[nkey]
                    while h >= 0:
                        if c_t[h] == 1 and c_run[h] >= 0.0 and ble[c_bend[h], 4] and (c_mask[h] & cm) == cm \
                                and c_h[h] <= chz and c_u[h] <= cu and c_g[h] <= ng + 1e-9:
                            dom = True
                            break
                        h = c_next[h]
                    if not dom:
                        if sid < 0:
                            sid = ctr[6]
                            ctr[6] = sid + 1
                            s_node[sid], s_dir[sid], s_run[sid], s_bend[sid], s_mask[sid], s_h[sid], s_u[sid] = \
                                cflat, cd, 0.0, 4, cm, chz, cu
                            s_t[sid] = 1
                            s_next[sid] = open_head[nkey]
                            open_head[nkey] = sid
                        s_g[sid] = ng
                        s_par[sid] = cur
                        dist = hdist[cflat]
                        if cd != end_dir:
                            bends = 1
                        elif dist == 0:
                            bends = 0
                        else:
                            v0, v1, v2 = goal_x - px[cflat], goal_y - py[cflat], goal_z - pz[cflat]
                            e0, e1, e2 = dir_vec[cd, 0], dir_vec[cd, 1], dir_vec[cd, 2]
                            tt = (0 + v0 * e0 + v1 * e1 + v2 * e2) / (e0 * e0 + e1 * e1 + e2 * e2)
                            on_ray = tt > 0 and abs(v0 - tt * e0) < 1e-6 and abs(v1 - tt * e1) < 1e-6 \
                                and abs(v2 - tt * e2) < 1e-6
                            bends = 0 if on_ray else 2
                        f = ng + (dist * kgmm + bends * e45)
                        pos = ctr[4]
                        ctr[4] = pos + 1
                        tie = ctr[2]
                        while pos > 0:
                            par = (pos - 1) >> 1
                            if f < hf[par] or (f == hf[par] and tie < ht[par]):
                                hf[pos], ht[pos], hg[pos], hid[pos] = hf[par], ht[par], hg[par], hid[par]
                                pos = par
                            else:
                                break
                        hf[pos], ht[pos], hg[pos], hid[pos] = f, tie, ng, sid
                        ctr[2] += 1
                        ctr[1] += 1
            for d in range(18):
                di = defl_idx[cd, d]
                if di < 0:
                    continue
                if di > 0 and crun < need_m[cb, di]:
                    continue
                if not ((cm >> di) & 1):     # D60: 들어온 엣지가 이 꺾임(끝 접선 깎기)을 허용하는가
                    continue
                nflat = step[cflat, d]
                if nflat < 0:
                    continue
                nm = (emask[cflat, d] >> (di * 4)) & 15
                if nm == 0:
                    continue
                if di > 0:
                    if not (clearance[cflat] >= thr[di] and pipe_margin[cflat] >= thr[di]):
                        k = (cflat * 18 + cd) * 18 + d
                        if k in arc:
                            ok_arc = arc[k]
                        else:
                            ok_arc = _arc_prefilter(G, cd, d, px[cflat], py[cflat], pz[cflat])
                            if ok_arc < 0:   # 정밀 판정 필요 → 그래프의 elbow_clear (파이썬) 를 그대로 부른다
                                with numba.objmode(res="int64"):
                                    res = _arc_callback_generic(k)
                                ok_arc = res
                            arc[k] = np.int8(ok_arc)
                        if ok_arc == 0:
                            continue
                length = elen[cflat, d]
                nhz = 0.0
                if gflag:   # D61·D63: 상향 금지, 하향이면 0, 수평이면 누적 ≤ L_max (gravity.step 과 같은 식)
                    kd = dkind[d]
                    if kd == 2:
                        continue
                    if kd == 0:
                        hraw = chz + length
                        if not (hraw <= lmax + 1e-6):
                            continue
                        nhz = _round6(hraw)
                # D80 ⑨: 지지 없이 온 정규화 길이 u (support_router.support_step 과 같은 식)
                pk = spack[cflat, d]
                sm = smaxd[d]
                pa = pk & 2097151
                cu2 = cu + bendu[di]
                if pa == 2097151:
                    nu = cu2 + length / sm
                    if not (nu <= 1 + 1e-9):
                        continue
                    nu = _round6(nu)
                else:
                    if not (cu2 + pa / sm <= 1 + 1e-9):
                        continue
                    if not (((pk >> 42) & 2097151) / sm <= 1 + 1e-9):
                        continue
                    nu = _round6(((pk >> 21) & 2097151) / sm)
                if di > 0:
                    run = length
                    nb = di
                else:
                    run = crun + length
                    nb = cb
                cap = run_cap[nb]
                run = _round6(run if run <= cap else cap)
                ng = g + (length / 1000 * kgpm + ekg[di] + scost[cflat, d])   # D57 추정 서포트 포함
                nkey = nflat * 18 + d
                sid = open_head[nkey]
                while sid >= 0:
                    if s_t[sid] == ct and s_run[sid] == run and s_bend[sid] == nb and s_mask[sid] == nm and s_h[sid] == nhz \
                            and s_u[sid] == nu:
                        break
                    sid = s_next[sid]
                if sid >= 0 and not (ng + 1e-9 < s_g[sid]):
                    continue
                dom = False
                h = closed_head[nkey]
                while h >= 0:
                    if c_t[h] == ct and c_run[h] >= run and ble[c_bend[h], nb] and (c_mask[h] & nm) == nm \
                            and c_h[h] <= nhz and c_u[h] <= nu and c_g[h] <= ng + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                if sid < 0:
                    sid = ctr[6]
                    ctr[6] = sid + 1
                    s_node[sid], s_dir[sid], s_run[sid], s_bend[sid], s_mask[sid], s_h[sid], s_u[sid] = nflat, d, run, nb, nm, nhz, nu
                    s_t[sid] = ct
                    s_next[sid] = open_head[nkey]
                    open_head[nkey] = sid
                s_g[sid] = ng
                s_par[sid] = cur
                # 휴리스틱 (router_astar.heuristic_factory 와 같은 식)
                dist = hdist[nflat]
                if d != end_dir:
                    bends = 1
                elif dist == 0:
                    bends = 0
                else:
                    v0, v1, v2 = goal_x - px[nflat], goal_y - py[nflat], goal_z - pz[nflat]
                    e0, e1, e2 = dir_vec[d, 0], dir_vec[d, 1], dir_vec[d, 2]
                    tt = (0 + v0 * e0 + v1 * e1 + v2 * e2) / (e0 * e0 + e1 * e1 + e2 * e2)
                    on_ray = tt > 0 and abs(v0 - tt * e0) < 1e-6 and abs(v1 - tt * e1) < 1e-6 \
                        and abs(v2 - tt * e2) < 1e-6
                    bends = 0 if on_ray else 2
                f = ng + (dist * kgmm + bends * e45)
                if hflag:   # D126: h = max(D41 h, 하한 표[노드]) — 표가 없으면 f 는 이전과 비트 동일
                    hb = hlow[nflat]
                    if hb > dist * kgmm + bends * e45:
                        f = ng + hb
                # push
                pos = ctr[4]
                ctr[4] = pos + 1
                tie = ctr[2]
                while pos > 0:
                    par = (pos - 1) >> 1
                    if f < hf[par] or (f == hf[par] and tie < ht[par]):
                        hf[pos], ht[pos], hg[pos], hid[pos] = hf[par], ht[par], hg[par], hid[par]
                        pos = par
                    else:
                        break
                hf[pos], ht[pos], hg[pos], hid[pos] = f, tie, ng, sid
                ctr[2] += 1
                ctr[1] += 1


