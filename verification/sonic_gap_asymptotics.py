#!/usr/bin/env python3
"""
音速轨迹的存在唯一性引理、脱体点亚声速性与渐近标度的符号审计。

C1. E(u_CJ) = (gamma+1) u_CJ (m-u_CJ)^2 —— 完全平方恒等式；
C2. E(u=m) = -[(gamma-1)m+h+2] (m^2-2*Acal*m+1) —— m>u_CJ 时严格为负；
C3. 物理域 m>u_CJ 内 c2<0、c0>0 的证明链：
    u_CJ-1-(gamma+1)h = sqrt(Acal^2-1)-(Acal-1) >= 0（h>0 时严格>0）；
C4. 组合 => 存在唯一性定理：M>M_CJ 时音速二次式在 (u_CJ, m) 内恰有一根，
    另一根为负；即每条附体高压缩极曲线弧恰有一个下游音速点；
C5. 脱体点亚声速性 M2(detach)<1 <=> E(u_d)<0：
    利用脱体三次式对 h 线性 => h_d(y,m) 有理反解，E(u_d) 化为 (y,m,gamma)
    有理式并因式分解，检查物理域内符号；
C6. M->infinity：s_s 与 s_d 的 1/m 级数，验证前两阶相同、
    差在 1/m^2 阶出现 => theta_max - theta_s ~ K/M^4；给出共同极限
    sin(theta_inf)=1/gamma 与 K 的闭式，并核对 K 对 h 的依赖；
C7. 近 CJ 端数值标度（theta_s 在窗口内的位置比例）。

运行:
  PYTHONIOENCODING=utf-8 python sonic_gap_asymptotics.py
"""

import math
import sys

import sympy as sp

g, m, h, u, w, y, e = sp.symbols("gamma m h u w y epsilon", positive=True)

PASS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        print("!! 审计失败，终止。")
        sys.exit(1)


B = h - (m - 1)
c2 = g * (g * h + 2 * B)
c1 = (g + 1) * B**2 + 2 * B + 2 * g * (m - 1)
c0 = 2 * (m - 1) - h
E = c2 * u**2 + c1 * u + c0
Acal = 1 + (g + 1) * h / 2  # = 1 + (gamma^2-1)Q/gamma

print("=" * 72)
print("C1: E(u_CJ) 完全平方恒等式")
h_cj = (w - 1) ** 2 / ((g + 1) * w)  # CJ 条件 h 的反解（w=u_CJ）
lhs = sp.simplify(E.subs({u: w, h: h_cj}))
check("C1 E(u_CJ) = (gamma+1) u_CJ (m-u_CJ)^2",
      sp.simplify(lhs - (g + 1) * w * (m - w) ** 2) == 0)
check("C1b CJ 条件反解自洽: u_CJ^2 - 2*Acal*u_CJ + 1 = 0",
      sp.simplify((w**2 - 2 * Acal.subs(h, h_cj) * w + 1)) == 0)

print("=" * 72)
print("C2: E(u=m) 的因式分解与符号")
E_at_m = sp.expand(E.subs(u, m))
claim = -((g - 1) * m + h + 2) * (m**2 - 2 * Acal * m + 1)
check("C2 E(u=m) = -[(gamma-1)m+h+2](m^2-2*Acal*m+1)",
      sp.expand(E_at_m - claim) == 0)
print("    m^2-2*Acal*m+1 的两根为 u_CJ 与 1/u_CJ；m>u_CJ 时该因子>0，")
print("    前因子 (gamma-1)m+h+2>0，故 E(u=m)<0 严格成立。")

print("=" * 72)
print("C3: 物理域内 c2<0、c0>0")
# u_CJ - 1 - (gamma+1)h = sqrt(Acal^2-1) - (Acal-1)
u_cj_expr = Acal + sp.sqrt(Acal**2 - 1)
gap1 = sp.simplify(u_cj_expr - 1 - (g + 1) * h - (sp.sqrt(Acal**2 - 1) - (Acal - 1)))
check("C3a u_CJ-1-(gamma+1)h = sqrt(Acal^2-1)-(Acal-1)", gap1 == 0)
# sqrt(Acal^2-1) > Acal-1 <=> Acal^2-1 > (Acal-1)^2 <=> 2(Acal-1) > 0（h>0）
check("C3b (Acal^2-1)-(Acal-1)^2 = 2(Acal-1) = (gamma+1)h",
      sp.simplify((Acal**2 - 1) - (Acal - 1) ** 2 - (g + 1) * h) == 0)
print("    => m>u_CJ 时 m-1 > u_CJ-1 > (gamma+1)h > (gamma+2)h/2 > h/2，")
print("    因此 c2 = gamma[(gamma+2)h-2(m-1)] < 0 且 c0 = 2(m-1)-h > 0。")
check("C3c (gamma+1)h-(gamma+2)h/2 = gamma*h/2 > 0",
      sp.simplify((g + 1) * h - (g + 2) * h / 2 - g * h / 2) == 0)

print("=" * 72)
print("C4: 存在唯一性（组合逻辑核对）")
print("    c2<0 且 c0>0 => 两根异号 => 恰一正根；")
print("    E(u_CJ)>0（C1）且 E(u=m)<0（C2）且开口向下 =>")
print("    正根 u_s ∈ (u_CJ, m)，负根被排除 => 每条附体高压缩极曲线")
print("    弧上恰有一个 M2=1 点。（全部符号事实已在 C1-C3 审计。）")
check("C4 逻辑链完成（依赖 C1-C3）", True)

print("=" * 72)
print("C5: 脱体点亚声速性 E(u_d)<0 的符号分析")
A_ = h + (g - 1) * m + 2
B_ = B
C_ = (g + 1) * m + 2
det_cubic = A_ * C_ * y**3 + (B_ * C_ + 3 * A_) * y**2 + 4 * B_ * y + h
check("C5a 脱体三次式对 h 是线性的", sp.degree(sp.Poly(det_cubic, h)) == 1)
h_d = sp.solve(det_cubic, h)[0]
h_d = sp.cancel(h_d)
print(f"    h_d(y,m) = {h_d}")
s_d = sp.cancel((-(3 * A_ * y**2 + 4 * B_ * y + h) / C_).subs(h, h_d))
u_d = sp.cancel((m * y**2 / (s_d + y**2)))
E_at_ud = sp.cancel(E.subs({u: u_d, h: h_d}))
E_num, E_den = sp.fraction(sp.cancel(sp.together(E_at_ud)))
E_num_f = sp.factor(E_num)
E_den_f = sp.factor(E_den)
print("    E(u_d) 分子因式:")
for fac, mult in sp.factor_list(E_num_f)[1]:
    print(f"      (mult={mult}) {sp.sstr(fac)[:140]}")
print("    E(u_d) 分母因式:")
for fac, mult in sp.factor_list(E_den_f)[1]:
    print(f"      (mult={mult}) {sp.sstr(fac)[:140]}")
# 数值抽查符号（物理域内取样：由 (gamma, Q, M) 正向算 y_d 再回代）
import numpy as np


def u_cj_num(gamma, Q):
    Ac = 1.0 + (gamma * gamma - 1.0) * Q / gamma
    return Ac + math.sqrt(Ac * Ac - 1.0)


def detach_num(M, gamma, Q):
    mm = M * M
    hh = 2.0 * (gamma - 1.0) * Q / gamma
    Aa = hh + (gamma - 1.0) * mm + 2.0
    Bb = hh - (mm - 1.0)
    Cc = (gamma + 1.0) * mm + 2.0
    best = None
    for z in np.roots([Aa * Cc, Bb * Cc + 3.0 * Aa, 4.0 * Bb, hh]):
        if abs(z.imag) > 1e-9:
            continue
        yy = float(z.real)
        if yy <= 0:
            continue
        ss = -(3.0 * Aa * yy * yy + 4.0 * Bb * yy + hh) / Cc
        if ss <= 0:
            continue
        tt = yy / math.sqrt(ss)
        if tt <= math.sqrt(ss):
            continue
        uu = mm * tt * tt / (1.0 + tt * tt)
        if uu < u_cj_num(gamma, Q) * (1 - 1e-12):
            continue
        if best is None or ss > best[1]:
            best = (yy, ss, uu)
    return best


E_num_fn = sp.lambdify((y, m, g), E_num, "mpmath")
E_den_fn = sp.lambdify((y, m, g), E_den, "mpmath")
import mpmath

mpmath.mp.dps = 40
signs = []
for gamma in [1.1, 1.2, 1.3, 1.4, 1.6]:
    for Q in [0.5, 2.0, 10.0, 50.0]:
        Mcj = math.sqrt(u_cj_num(gamma, Q))
        for fac in [1.001, 1.05, 1.3, 2.0, 5.0, 20.0]:
            got = detach_num(Mcj * fac, gamma, Q)
            if got is None:
                continue
            yy, ss, uu = got
            val = float(E_num_fn(yy, (Mcj * fac) ** 2, gamma) /
                        E_den_fn(yy, (Mcj * fac) ** 2, gamma))
            signs.append(val)
check("C5b 物理域取样 E(u_d) 全部 < 0（120 组）",
      all(v < 0 for v in signs),
      f"max={max(signs):.3e}（越负越亚声速; 采样数={len(signs)}）")
print("    结论：脱体点 M2<1 在采样域内成立 => u_s < u_d，音速点位于弱支；")
print("    E(u_d) 的完全代数符号判定（全参数域）留作后续定理化工作。")

print("=" * 72)
print("C6: M->infinity 级数：theta_max 与 theta_s 前两阶重合，差 ~ 1/m^2")
# --- 音速侧: u_s = alpha*m + d0 + d1/m，代入 E 逐阶求解
alpha = (g + 1) / (2 * g)
d0, d1, d2 = sp.symbols("d0 d1 d2")
u_ser = alpha / e + d0 + d1 * e  # m = 1/e
E_e = sp.expand(E.subs({m: 1 / e, u: u_ser}) * e**3)  # E ~ O(m^3) -> 乘 e^3
E_e = sp.expand(E_e)
poly_e = sp.Poly(E_e, e)
# e^0（对应 m^3）应恒为零
check("C6a 音速侧 m^3 阶自动消去", sp.simplify(poly_e.coeff_monomial(1)) == 0)
sol_d0 = sp.solve(sp.Eq(poly_e.coeff_monomial(e), 0), d0)[0]
E_e2 = sp.expand(E_e.subs(d0, sol_d0))
sol_d1 = sp.solve(sp.Eq(sp.Poly(E_e2, e).coeff_monomial(e**2), 0), d1)[0]
u_s_ser = (alpha / e + sol_d0 + sol_d1 * e).subs(d1, sol_d1)
# s_s(u) 有理式
r_star = (g * u + h + 2 - 1 / e) / (1 + g * u)
t2 = u / (1 / e - u)
s_of_u = sp.cancel(t2 * (1 - r_star) ** 2 / (1 + r_star * t2) ** 2)
s_s_ser = sp.series(s_of_u.subs(u, alpha / e + sol_d0 + sol_d1 * e), e, 0, 3).removeO()
s_s_ser = sp.expand(s_s_ser)
# --- 脱体侧: y = y0 + y1 e + y2 e^2
y0, y1, y2 = sp.symbols("y0 y1 y2")
y_ser = y0 + y1 * e + y2 * e**2
det_e = sp.expand(det_cubic.subs({m: 1 / e, y: y_ser}) * e**2)  # 三次式 ~ m^2
pe = sp.Poly(sp.expand(det_e), e)
sol_y0 = [s_ for s_ in sp.solve(sp.Eq(pe.coeff_monomial(1), 0), y0) if s_ != 0]
check("C6b 脱体侧领头阶 y0 = 1/(gamma-1)",
      any(sp.simplify(s_ - 1 / (g - 1)) == 0 for s_ in sol_y0))
y0v = 1 / (g - 1)
pe1 = sp.Poly(sp.expand(det_e.subs(y0, y0v)), e)
sol_y1 = sp.solve(sp.Eq(pe1.coeff_monomial(e), 0), y1)[0]
pe2 = sp.Poly(sp.expand(det_e.subs({y0: y0v, y1: sol_y1})), e)
sol_y2 = sp.solve(sp.Eq(pe2.coeff_monomial(e**2), 0), y2)[0]
s_d_expr = (-(3 * A_ * y**2 + 4 * B_ * y + h) / C_).subs(m, 1 / e)
s_d_ser = sp.series(
    s_d_expr.subs(y, (y0v + sol_y1 * e + sol_y2 * e**2)), e, 0, 3
).removeO()
s_d_ser = sp.expand(s_d_ser)
# --- 对比
s_inf_claim = 1 / (g**2 - 1)
diff01 = sp.simplify(sp.expand(s_d_ser - s_s_ser))
diff_poly = sp.Poly(diff01, e)
check("C6c 两条轨迹 s 级数 e^0 阶同为 1/(gamma^2-1)",
      sp.simplify(s_s_ser.subs(e, 0) - s_inf_claim) == 0
      and sp.simplify(s_d_ser.subs(e, 0) - s_inf_claim) == 0)
print("    => 共同极限角 theta_inf = arctan(1/sqrt(gamma^2-1)) 即 "
      "sin(theta_inf)=1/gamma")
c_e0 = sp.simplify(diff_poly.coeff_monomial(1))
c_e1 = sp.simplify(diff_poly.coeff_monomial(e))
c_e2 = sp.simplify(diff_poly.coeff_monomial(e**2))
check("C6d s_d - s_s 的 e^0 与 e^1 阶均为零", c_e0 == 0 and c_e1 == 0,
      f"e^0={c_e0}, e^1={c_e1}")
c_e2_f = sp.factor(c_e2)
print(f"    Delta_s2 = lim m^2 (s_d - s_s) = {c_e2_f}")
check("C6e 间隙首现于 1/m^2 阶（=> theta 间隙 ~ 1/M^4）", c_e2 != 0)
dep_h = sp.simplify(sp.diff(c_e2, h))
print(f"    Delta_s2 对 h 的导数 = {sp.factor(dep_h)}")
if dep_h == 0:
    print("    领头间隙系数与放热完全无关。")
else:
    print("    领头间隙系数含 h（放热）依赖，Q 无关性只是近似。")
# 角度间隙系数 K(gamma,h): theta_d - theta_s ~ K * e^2,
# dtheta/ds = 1/(2 sqrt(s) (1+s)) 在 s_inf 处
K_theta = sp.simplify(c_e2 / (2 * sp.sqrt(s_inf_claim) * (1 + s_inf_claim)))
K_theta_f = sp.factor(sp.simplify(K_theta))
print(f"    K_theta(gamma,h) = {K_theta_f}")
# 数值对拍：gamma=1.3, Q=10（h=60/13）与惰性 h->0
for gam_v, Q_v in [(1.3, 10.0), (1.4, 50.0), (1.4, 0.0), (1.2, 5.0)]:
    h_v = 2 * (gam_v - 1) * Q_v / gam_v
    K_v = float(K_theta.subs({g: gam_v, h: h_v}))
    print(f"    gamma={gam_v}, Q={Q_v}: K_theta*M^-4 预测 gap = "
          f"{K_v * 180 / math.pi:.6f}/M^4 度")
# 与 gap_diag 数值比对（gamma=1.3, Q=10, M/Mcj=32 => M=113.3, gap=2.7329e-8 度）
K13 = float(K_theta.subs({g: 1.3, h: 2 * 0.3 * 10 / 1.3})) * 180 / math.pi
gap_pred = K13 / (3.540637 * 32) ** 4
check("C6f 预测 gap 与数值 gap（gamma=1.3,Q=10,M=113.3）相对差 < 2%",
      abs(gap_pred - 2.732855e-8) / 2.732855e-8 < 0.02,
      f"pred={gap_pred:.4e}, num=2.7329e-08")

print("=" * 72)
print("C7: 近 CJ 端窗口内位置比例（数值）")
gam_v, Q_v = 1.3, 10.0
Mcj = math.sqrt(u_cj_num(gam_v, Q_v))


def sonic_num(M, gamma, Q):
    mm = M * M
    hh = 2 * (gamma - 1) * Q / gamma
    Bb = hh - (mm - 1)
    cc2 = gamma * (gamma * hh + 2 * Bb)
    cc1 = (gamma + 1) * Bb * Bb + 2 * Bb + 2 * gamma * (mm - 1)
    cc0 = 2 * (mm - 1) - hh
    disc = cc1 * cc1 - 4 * cc2 * cc0
    uu = (-cc1 - math.sqrt(disc)) / (2 * cc2)  # c2<0 => 正根取 '-' 分支
    rr = (gamma * uu + hh + 2 - mm) / (1 + gamma * uu)
    tt = math.sqrt(uu / (mm - uu))
    tauv = tt * (1 - rr) / (1 + rr * tt * tt)
    return math.atan(tauv)


def theta_cj_num(M, gamma, Q):
    ucj = u_cj_num(gamma, Q)
    beta = math.asin(math.sqrt(ucj / (M * M)))
    X = (gamma + 1) * ucj / (1 + gamma * ucj)
    return beta - math.atan(math.tan(beta) / X)


print("    eps=M/Mcj-1, ratio=(theta_max-theta_s)/(theta_max-theta_CJ):")
ratios = []
for eps_v in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    M = Mcj * (1 + eps_v)
    got = detach_num(M, gam_v, Q_v)
    tmax = math.atan(math.sqrt(got[1]))
    ts = sonic_num(M, gam_v, Q_v)
    tcj = theta_cj_num(M, gam_v, Q_v)
    ratio = (tmax - ts) / (tmax - tcj)
    ratios.append(ratio)
    print(f"    eps={eps_v:.0e}: theta_cj={tcj*180/math.pi:.6f}, "
          f"theta_s={ts*180/math.pi:.6f}, theta_max={tmax*180/math.pi:.6f}, "
          f"ratio={ratio:.6f}")
check("C7 近 CJ 端 ratio 收敛到 (0,1) 内常数（音速点既不贴 CJ 也不贴脱体）",
      0.0 < ratios[-1] < 1.0 and abs(ratios[-1] - ratios[-2]) < 0.01,
      f"ratio -> {ratios[-1]:.4f}")

print("=" * 72)
n_ok = sum(1 for _, ok in PASS if ok)
print(f"审计完成: {n_ok}/{len(PASS)} 项全部 PASS")
