#!/usr/bin/env python3
"""
音速轨迹（波后 M2=1）闭式解的数值验证与参数域扫描。

对照对象：
1. 锚点参数 gamma=1.3, M=7, Q=10 的已封牌端点
   （beta_CJ=30.384, theta_CJ=11.006, theta_max=40.3829, beta_max=67.4017 度）；
2. 音速二次式 E(u)=c2 u^2+c1 u+c0 的根 vs 在原始含根号高压缩极曲线上
   直接对 M2(beta)-1 做 Brent 求根（完全独立路线）；
3. 分支归属：beta_s 与 beta_max 的相对位置（弱/强支判定）、
   强支全段 M2<1、低压缩支全段 M2>1 的网格核查；
4. (M,theta) 平面三条轨迹 theta_CJ(M) < theta_s(M) < theta_max(M)
   的排序与不相交性（多 gamma、多 Q 扫描）；
5. M->infinity 渐近式 u_s ~ (gamma+1)m/(2gamma) + delta* 的收敛检查；
6. A12 隐式二次因子在数值点上的残差。

运行:
  PYTHONIOENCODING=utf-8 python sonic_numerical_check.py
输出: results_sonic/ 下 CSV + 报告 + 图。
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

RAD = 180.0 / math.pi
OUT = Path(__file__).parent / "results_sonic"
OUT.mkdir(exist_ok=True)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- 基本量

def qprime(gamma: float, Q: float) -> float:
    return 2.0 * (gamma - 1.0) * Q / gamma


def u_cj_of(gamma: float, Q: float) -> float:
    Acal = 1.0 + (gamma * gamma - 1.0) * Q / gamma
    return Acal + math.sqrt(Acal * Acal - 1.0)


def cj_point(M: float, gamma: float, Q: float) -> tuple[float, float]:
    """返回 (beta_CJ, theta_CJ)，弧度。"""
    u_cj = u_cj_of(gamma, Q)
    m = M * M
    beta = math.asin(math.sqrt(u_cj / m))
    X = (gamma + 1.0) * u_cj / (1.0 + gamma * u_cj)
    theta = beta - math.atan(math.tan(beta) / X)
    return beta, theta


def real_cubic_roots(a: float, b: float, c: float, d: float) -> list[float]:
    """实系数三次方程全部实根（numpy.roots 后筛实根，仅用于对照用途）。"""
    roots = np.roots([a, b, c, d])
    return [float(z.real) for z in roots if abs(z.imag) < 1e-9 * max(1.0, abs(z))]


def detachment(M: float, gamma: float, Q: float) -> tuple[float, float, float]:
    """脱体点 (theta_max, beta_max, u_d)，用重根 y 三次式。"""
    m = M * M
    h = qprime(gamma, Q)
    A = h + (gamma - 1.0) * m + 2.0
    B = h - (m - 1.0)
    C = (gamma + 1.0) * m + 2.0
    u_cj = u_cj_of(gamma, Q)
    best = None
    for y in real_cubic_roots(A * C, B * C + 3.0 * A, 4.0 * B, h):
        if y <= 0.0:
            continue
        s = -(3.0 * A * y * y + 4.0 * B * y + h) / C
        if s <= 0.0:
            continue
        tau = math.sqrt(s)
        t = y / tau
        if t <= tau:
            continue
        u = m * t * t / (1.0 + t * t)
        if u < u_cj * (1.0 - 1e-12):
            continue
        r = (t - tau) / (t * (1.0 + t * tau))
        R = 1.0 + gamma * u - (gamma + 1.0) * u * r
        if R < -1e-9:
            continue
        if best is None or s > best[0]:
            best = (s, y, u)
    if best is None:
        raise RuntimeError("脱体三次式无合法根")
    s, y, u = best
    return math.atan(math.sqrt(s)), math.atan(y / math.sqrt(s)), u


# ------------------------------------------------- 高/低压缩支上的 M2(beta)

def branch_state(beta: float, M: float, gamma: float, Q: float, high: bool):
    """原始含根号 R-H 支上的 (r, P, M2sq, theta)。不经过任何多项式。"""
    m = M * M
    u = m * math.sin(beta) ** 2
    D = (u - 1.0) ** 2 - 2.0 * (gamma * gamma - 1.0) * u * Q / gamma
    if D < 0.0:
        if D > -1e-10 * max(1.0, (u - 1.0) ** 2):
            D = 0.0
        else:
            return None
    sq = math.sqrt(D)
    r = (1.0 + gamma * u + (-sq if high else sq)) / ((gamma + 1.0) * u)
    P = 1.0 + gamma * u * (1.0 - r)
    M2sq = (u * r * r + m - u) / (P * r)
    t = math.tan(beta)
    tau = t * (1.0 - r) / (1.0 + r * t * t)
    return r, P, M2sq, math.atan(tau)


# ------------------------------------------------------- 音速二次式闭式解

def sonic_coefficients(m: float, gamma: float, h: float) -> tuple[float, float, float]:
    B = h - (m - 1.0)
    c2 = gamma * (gamma * h + 2.0 * B)
    c1 = (gamma + 1.0) * B * B + 2.0 * B + 2.0 * gamma * (m - 1.0)
    c0 = 2.0 * (m - 1.0) - h
    return c2, c1, c0


def sonic_points(M: float, gamma: float, Q: float):
    """
    解音速二次式并做物理筛选。
    返回列表[dict]，含 u, beta, theta, r, R, 支身份。
    """
    m = M * M
    h = qprime(gamma, Q)
    u_cj = u_cj_of(gamma, Q)
    c2, c1, c0 = sonic_coefficients(m, gamma, h)
    scale = max(abs(c2), abs(c1), abs(c0))
    roots: list[float] = []
    if abs(c2) < 1e-14 * scale:
        if abs(c1) > 0.0:
            roots = [-c0 / c1]
    else:
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc >= 0.0:
            sq = math.sqrt(disc)
            roots = [(-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)]
    out = []
    for u in roots:
        if not (u_cj - 1e-9 <= u <= m * (1.0 + 1e-12)):
            continue
        r = (gamma * u + h + 2.0 - m) / (1.0 + gamma * u)
        if r <= 0.0:
            continue
        R = 1.0 + gamma * u - (gamma + 1.0) * u * r
        if u >= m:  # beta=90 度退化端点
            t = math.inf
            tau = 0.0
        else:
            t = math.sqrt(u / (m - u))
            tau = t * (1.0 - r) / (1.0 + r * t * t)
        if tau < 0.0 or (t is not math.inf and t <= tau):
            continue
        out.append(
            dict(u=u, r=r, R=R,
                 beta=math.atan(t) if t is not math.inf else math.pi / 2,
                 theta=math.atan(tau),
                 branch=("high" if R >= 0.0 else "low"))
        )
    return out


# ============================================================ 1. 锚点核对
print("=" * 72)
print("1. 锚点参数 gamma=1.3, M=7, Q=10")
GA, MA, QA = 1.3, 7.0, 10.0
beta_cj, theta_cj = cj_point(MA, GA, QA)
theta_max, beta_max, u_d = detachment(MA, GA, QA)
check("锚点 beta_CJ = 30.384 度", abs(beta_cj * RAD - 30.384) < 5e-4,
      f"{beta_cj * RAD:.6f}")
check("锚点 theta_CJ = 11.006 度（交接值系 11.0055 的进位舍入）",
      abs(theta_cj * RAD - 11.0055) < 1e-3, f"{theta_cj * RAD:.6f}")
check("锚点 theta_max = 40.3829 度", abs(theta_max * RAD - 40.3829) < 5e-5,
      f"{theta_max * RAD:.6f}")
check("锚点 beta_max = 67.4017 度", abs(beta_max * RAD - 67.4017) < 5e-5,
      f"{beta_max * RAD:.6f}")

# ============================================ 2. 二次式闭式 vs Brent 独立求根
print("=" * 72)
print("2. 音速点：二次式闭式 vs 原始极曲线 Brent")
sp_list = sonic_points(MA, GA, QA)
check("锚点参数恰有 1 个合法音速点", len(sp_list) == 1,
      f"count={len(sp_list)}")
sonic = sp_list[0]
print(f"    u_s={sonic['u']:.12f}, beta_s={sonic['beta'] * RAD:.8f} 度, "
      f"theta_s={sonic['theta'] * RAD:.8f} 度, R={sonic['R']:.6f} ({sonic['branch']})")

# 独立路线：beta 网格上找 M2(beta)-1 的全部变号，Brent 收敛
grid = np.linspace(beta_cj + 1e-9, math.pi / 2 - 1e-12, 40001)
vals = []
for b in grid:
    st = branch_state(float(b), MA, GA, QA, high=True)
    vals.append(st[2] - 1.0 if st else math.nan)
vals = np.asarray(vals)
crossings = []
for i in range(len(grid) - 1):
    if math.isnan(vals[i]) or math.isnan(vals[i + 1]):
        continue
    if vals[i] == 0.0:
        crossings.append(float(grid[i]))
    elif vals[i] * vals[i + 1] < 0.0:
        f = lambda b: branch_state(b, MA, GA, QA, high=True)[2] - 1.0
        crossings.append(float(brentq(f, float(grid[i]), float(grid[i + 1]),
                                      xtol=1e-15, maxiter=200)))
check("高压缩支全段恰有 1 个音速穿越", len(crossings) == 1,
      f"count={len(crossings)}")
beta_brent = crossings[0]
err_deg = abs(beta_brent - sonic["beta"]) * RAD
check("闭式 beta_s 与 Brent 之差 < 1e-9 度", err_deg < 1e-9,
      f"err={err_deg:.3e} 度")
st = branch_state(beta_brent, MA, GA, QA, high=True)
err_th = abs(st[3] - sonic["theta"]) * RAD
check("闭式 theta_s 与 Brent 之差 < 1e-9 度", err_th < 1e-9,
      f"err={err_th:.3e} 度")

# ================================================== 3. 分支归属与因果分区
print("=" * 72)
print("3. 分支归属与因果分区（锚点参数）")
check("音速点在高压缩支 (R>0)", sonic["R"] > 0.0, f"R={sonic['R']:.4f}")
check("音速点在弱支：beta_s < beta_max", sonic["beta"] < beta_max,
      f"beta_s={sonic['beta'] * RAD:.6f} < beta_max={beta_max * RAD:.6f}")
check("theta_CJ < theta_s < theta_max",
      theta_cj < sonic["theta"] < theta_max,
      f"{theta_cj * RAD:.4f} < {sonic['theta'] * RAD:.6f} < {theta_max * RAD:.6f}")
gap_deg = (theta_max - sonic["theta"]) * RAD
print(f"    亚声速楔角窗口宽 theta_max - theta_s = {gap_deg:.6f} 度")
print(f"    beta_max - beta_s = {(beta_max - sonic['beta']) * RAD:.6f} 度")

# 脱体点处 M2 应 < 1（音速点在弱支内 => 合并点已经亚声速）
st_d = branch_state(beta_max, MA, GA, QA, high=True)
check("脱体点处 M2 < 1", st_d[2] < 1.0, f"M2={math.sqrt(st_d[2]):.6f}")

# CJ 点处 M2n=1 但 M2 > 1
st_cj = branch_state(beta_cj, MA, GA, QA, high=True)
u_cj = u_cj_of(GA, QA)
r_cj = (1.0 + GA * u_cj) / ((GA + 1.0) * u_cj)
P_cj = 1.0 + GA * u_cj * (1.0 - r_cj)
M2n_cj = math.sqrt(u_cj * r_cj / P_cj)
check("CJ 点 M2n = 1", abs(M2n_cj - 1.0) < 1e-12, f"M2n={M2n_cj:.14f}")
check("CJ 点 M2 > 1（切向分量）", st_cj[2] > 1.0,
      f"M2={math.sqrt(st_cj[2]):.6f}")

# 强支全段亚声速
strong = np.linspace(beta_max + 1e-9, math.pi / 2 - 1e-12, 5001)
m2s = [branch_state(float(b), MA, GA, QA, high=True)[2] for b in strong]
check("强支全段 M2 < 1", max(m2s) < 1.0, f"max M2={math.sqrt(max(m2s)):.6f}")

# 低压缩支全段超声速（除正 CJ 退化点）
low = np.linspace(beta_cj + 1e-9, math.pi / 2 - 1e-9, 5001)
m2l = [branch_state(float(b), MA, GA, QA, high=False)[2] for b in low]
check("低压缩支全段 M2 > 1", min(m2l) > 1.0, f"min M2={math.sqrt(min(m2l)):.6f}")

# ====================================== 4. (M,theta) 平面扫描：排序与相交性
print("=" * 72)
print("4. 参数域扫描：theta_CJ < theta_s < theta_max 与根计数")
rows = []
bad = 0
for gamma, Q in itertools.product([1.15, 1.2, 1.3, 1.4], [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]):
    M_cj = math.sqrt(u_cj_of(gamma, Q))
    for fac in [1.0005, 1.001, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]:
        M = M_cj * fac
        m = M * M
        h = qprime(gamma, Q)
        try:
            b_cj, t_cj = cj_point(M, gamma, Q)
            t_max, b_max, _ = detachment(M, gamma, Q)
        except Exception as e:
            bad += 1
            continue
        pts = sonic_points(M, gamma, Q)
        c2, _, _ = sonic_coefficients(m, gamma, h)
        n_high = sum(1 for p in pts if p["branch"] == "high")
        rec = dict(gamma=gamma, Q=Q, M=M, M_over_Mcj=fac, c2=c2,
                   n_sonic=len(pts), n_sonic_high=n_high,
                   theta_cj_deg=t_cj * RAD, theta_max_deg=t_max * RAD,
                   beta_max_deg=b_max * RAD)
        if pts:
            p0 = max(pts, key=lambda p: p["u"]) if len(pts) > 1 else pts[0]
            rec.update(theta_s_deg=p0["theta"] * RAD, beta_s_deg=p0["beta"] * RAD,
                       u_s=p0["u"], R_s=p0["R"],
                       order_ok=(t_cj < p0["theta"] < t_max),
                       weak_ok=(p0["beta"] < b_max))
        rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(OUT / "sonic_scan.csv", index=False, encoding="utf-8-sig")
n_total = len(df)
n_with = int((df["n_sonic_high"] >= 1).sum())
check("扫描全部参数组均恰有 1 个高压缩支音速点",
      bool((df["n_sonic_high"] == 1).all()),
      f"{n_with}/{n_total} 组有音速点; 多根组数={int((df['n_sonic_high']>1).sum())}")
sub = df.dropna(subset=["theta_s_deg"])
check("扫描全域 theta_CJ < theta_s < theta_max", bool(sub["order_ok"].all()),
      f"违例 {int((~sub['order_ok']).sum())}/{len(sub)}")
check("扫描全域音速点在弱支 (beta_s < beta_max)", bool(sub["weak_ok"].all()),
      f"违例 {int((~sub['weak_ok']).sum())}/{len(sub)}")
gap_rel = (sub["theta_max_deg"] - sub["theta_s_deg"])
print(f"    theta_max - theta_s: min={gap_rel.min():.6f} 度 "
      f"(gamma={sub.loc[gap_rel.idxmin(),'gamma']}, Q={sub.loc[gap_rel.idxmin(),'Q']}, "
      f"M/M_CJ={sub.loc[gap_rel.idxmin(),'M_over_Mcj']}), max={gap_rel.max():.4f} 度")
print(f"    c2 符号: 正 {int((sub['c2']>0).sum())} 组 / 负 {int((sub['c2']<0).sum())} 组"
      f"（两种符号下二次式均只给 1 个物理根）")

# ============================================ 5. M -> infinity 渐近收敛
print("=" * 72)
print("5. 渐近检查 u_s = (gamma+1)m/(2gamma) + delta* + O(1/m)")
asym_rows = []
for gamma, Q in [(1.2, 5.0), (1.3, 10.0), (1.4, 20.0)]:
    h = qprime(gamma, Q)
    delta_star = (gamma * (gamma * h - h + 2.0) - 2.0 * h - 6.0) / (4.0 * gamma)
    for M in [20.0, 50.0, 100.0, 200.0]:
        m = M * M
        pts = sonic_points(M, gamma, Q)
        u_s = max(p["u"] for p in pts)
        u_asym = (gamma + 1.0) / (2.0 * gamma) * m + delta_star
        asym_rows.append(dict(gamma=gamma, Q=Q, M=M, u_s=u_s,
                              u_asym=u_asym, diff=u_s - u_asym))
adf = pd.DataFrame(asym_rows)
adf.to_csv(OUT / "sonic_asymptotics.csv", index=False, encoding="utf-8-sig")
# 残差应随 m 增大衰减（O(1/m)）
ok_decay = True
for (gamma, Q), gdf in adf.groupby(["gamma", "Q"]):
    d = gdf.sort_values("M")["diff"].abs().to_numpy()
    ok_decay &= bool(np.all(np.diff(d) < 0.0))
check("渐近残差随 M 单调衰减（O(1/m) 收敛）", ok_decay,
      f"M=200 时最大 |残差|={adf[adf['M']==200]['diff'].abs().max():.3e}")
for gamma in [1.2, 1.3, 1.4]:
    print(f"    gamma={gamma}: beta_s(M->inf) -> "
          f"{math.asin(math.sqrt((gamma+1)/(2*gamma))) * RAD:.4f} 度（与 Q 无关）")

# ============================================ 6. 隐式二次因子数值残差
print("=" * 72)
print("6. (tan^2 theta, m) 隐式二次因子零点核对")
import sympy as sym

gs, ms, hs, us, ss = sym.symbols("gamma m h u s", positive=True)
Bs = hs - (ms - 1)
E_s = gs * (gs * hs + 2 * Bs) * us**2 + ((gs + 1) * Bs**2 + 2 * Bs + 2 * gs * (ms - 1)) * us + 2 * (ms - 1) - hs
r_s = (gs * us + hs + 2 - ms) / (1 + gs * us)
t2_s = us / (ms - us)
s_expr = sym.cancel(t2_s * (1 - r_s) ** 2 / (1 + r_s * t2_s) ** 2)
s_num, s_den = sym.fraction(s_expr)
res = sym.resultant(sym.Poly(E_s, us), sym.Poly(sym.expand(ss * s_den - s_num), us))
fac_list = [f for f, _ in sym.factor_list(res)[1] if f.has(ss)]
assert len(fac_list) == 1
implicit = sym.lambdify((ss, ms, gs, hs), fac_list[0], "numpy")
resid = []
for _, row in sub.sample(min(60, len(sub)), random_state=0).iterrows():
    s_val = math.tan(row["theta_s_deg"] / RAD) ** 2
    m_val = row["M"] ** 2
    h_val = qprime(row["gamma"], row["Q"])
    # 归一化残差
    raw = implicit(s_val, m_val, row["gamma"], h_val)
    scale = abs(implicit(s_val * 1.1 + 1e-3, m_val, row["gamma"], h_val)) + abs(raw) + 1.0
    resid.append(abs(raw) / scale)
check("隐式二次因子在音速点上的归一化残差 < 1e-8", max(resid) < 1e-8,
      f"max={max(resid):.3e}")

# ============================================ 7. 图
print("=" * 72)
print("7. 绘图")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

# (a) 锚点极曲线 theta-beta，标 CJ/音速/脱体，弱支亚声速段加粗
betas = np.linspace(beta_cj, math.pi / 2 - 1e-9, 2000)
th_hi = [branch_state(float(b), MA, GA, QA, high=True)[3] * RAD for b in betas]
m2_hi = [math.sqrt(branch_state(float(b), MA, GA, QA, high=True)[2]) for b in betas]
th_lo = [branch_state(float(b), MA, GA, QA, high=False)[3] * RAD for b in betas]
ax = axes[0]
bb = betas * RAD
sup = np.asarray(m2_hi) > 1.0
ax.plot(np.asarray(th_hi)[sup], bb[sup], "b-", lw=1.6,
        label="high-compression, $M_2>1$")
ax.plot(np.asarray(th_hi)[~sup], bb[~sup], "r-", lw=2.4,
        label="high-compression, $M_2<1$")
ax.plot(th_lo, bb, "g--", lw=1.0, alpha=0.7,
        label="low-compression ($M_2>1$)")
ax.plot([theta_cj * RAD], [beta_cj * RAD], "ko", ms=6)
ax.annotate("CJ", (theta_cj * RAD, beta_cj * RAD), textcoords="offset points",
            xytext=(6, -10))
ax.plot([sonic["theta"] * RAD], [sonic["beta"] * RAD], "r*", ms=14)
ax.annotate(r"$M_2=1$", (sonic["theta"] * RAD, sonic["beta"] * RAD),
            textcoords="offset points", xytext=(-58, 4))
ax.plot([theta_max * RAD], [beta_max * RAD], "ks", ms=6)
ax.annotate("detach", (theta_max * RAD, beta_max * RAD),
            textcoords="offset points", xytext=(-58, 8))
ax.set_xlabel(r"$\theta$ (deg)")
ax.set_ylabel(r"$\beta$ (deg)")
ax.set_title(rf"$\gamma$={GA}, $M$={MA:g}, $Q$={QA:g}: sonic point on the polar")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# (b) (M, theta) 平面三条轨迹, gamma=1.3, Q=10
ax = axes[1]
M_cj0 = math.sqrt(u_cj_of(GA, QA))
Ms = np.linspace(M_cj0 * 1.0004, 16.0, 400)
tcjs, tss, tmaxs = [], [], []
for M in Ms:
    _, tc = cj_point(float(M), GA, QA)
    tm, _, _ = detachment(float(M), GA, QA)
    pts = sonic_points(float(M), GA, QA)
    ts = max(p["theta"] for p in pts) if pts else math.nan
    tcjs.append(tc * RAD)
    tss.append(ts * RAD)
    tmaxs.append(tm * RAD)
ax.plot(Ms, tcjs, "k-", lw=1.3, label=r"$\theta_{CJ}(M)$")
ax.plot(Ms, tss, "r-", lw=1.6, label=r"$\theta_{s}(M)$  ($M_2=1$)")
ax.plot(Ms, tmaxs, "b-", lw=1.3, label=r"$\theta_{\max}(M)$")
ax.fill_between(Ms, tss, tmaxs, color="red", alpha=0.12,
                label=r"weak branch, $M_2<1$")
ax.axvline(M_cj0, color="gray", ls=":", lw=1)
ax.annotate(r"$M_{CJ}$", (M_cj0, ax.get_ylim()[1] * 0.05),
            textcoords="offset points", xytext=(4, 0), color="gray")
ax.set_xlabel(r"$M$")
ax.set_ylabel(r"$\theta$ (deg)")
ax.set_title(rf"$\gamma$={GA}, $Q$={QA:g}: three critical loci")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

fig.savefig(OUT / "sonic_locus.png", dpi=200)
plt.close(fig)
print(f"    图已保存: {OUT / 'sonic_locus.png'}")

# ============================================ 汇总
print("=" * 72)
n_ok = sum(1 for _, ok, _ in CHECKS if ok)
verdict = "PASS" if n_ok == len(CHECKS) else "FAIL"
print(f"总判定: {verdict}  ({n_ok}/{len(CHECKS)})")

report = [f"# 音速轨迹数值验证报告", "",
          f"总判定: {verdict} ({n_ok}/{len(CHECKS)})", "",
          "| 检查 | 结果 | 备注 |", "|---|---|---|"]
report += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in CHECKS]
report += ["", "关键数值（gamma=1.3, M=7, Q=10）：",
           f"- u_s = {sonic['u']:.12f}",
           f"- beta_s = {sonic['beta'] * RAD:.8f} 度",
           f"- theta_s = {sonic['theta'] * RAD:.8f} 度",
           f"- theta_max - theta_s = {gap_deg:.6f} 度",
           f"- 脱体点 M2 = {math.sqrt(st_d[2]):.6f}",
           f"- CJ 点 M2 = {math.sqrt(st_cj[2]):.6f}",
           "", "输出: sonic_scan.csv, sonic_asymptotics.csv, sonic_locus.png"]
(OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
print(f"报告: {OUT / 'report.md'}")
raise SystemExit(0 if verdict == "PASS" else 1)
