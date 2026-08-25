#!/usr/bin/env python3
"""
音速轨迹（波后 M2=1）代数化的符号审计。

审计对象（全部为待验证断言，未通过则脚本报错退出）：

A1. 波后总马赫数恒等式 M2^2 = (u r^2 + m - u) / (P r)，
    其中 P = 1 + gamma*u*(1-r)，由原始速度分量与 T2/T1 = P*r 直接推出；
A2. M2^2 = M2n^2 + (m-u)/(P r)，其中 M2n^2 = u r / P（法向+切向分解）；
A3. M2=1 等价于二次型 S(r) = (gamma+1) u r^2 - (1+gamma*u) r + (m-u) = 0；
A4. S 与 R-H 主方程 RH(r) = (gamma+1)u r^2 - 2(1+gamma u) r + (gamma-1)u + 2 + h
    线性消去 r^2 后得 r* = (gamma*u + h + 2 - m) / (1 + gamma*u)；
A5. 把 r* 代回 RH 并乘 (1+gamma*u)^2 后，u^3 系数恒为零，
    音速轨迹是 u 的二次方程 E(u) = c2 u^2 + c1 u + c0 = 0，
    c2 = gamma*(gamma*h + 2B), c1 = (gamma+1)B^2 + 2B + 2 gamma (m-1),
    c0 = 2(m-1) - h, 其中 B = h - (m-1)；
A6. E 与 Res_r(RH, S) 只差非零常数因子（消元路线一致性）；
A7. beta=90 度极限（m=u）：E(u)|_{m=u} 以 u=u_CJ 为根
    （音速轨迹的正波端点 = 正 CJ 波）；
A8. 惰性极限 h=0：E 含因子 (m-1-u)... 实际验证 h=0 时 E/(2(m-1)) 与
    经典惰性音速条件 gamma u^2 - [ (gamma+1)(m-1)/2 + ... ] 的一致性
    （以多项式相等方式验证，不引用外部公式，而是从惰性 R-H 独立重建）；
A9. 低压缩支恒超声速：M2^2 - 1 = (S 的符号) / (P r)，且在低压缩支
    r = r_L 处 S < 0 ... 以符号方式验证 S(r) 在两 R-H 根处的取值
    S(r_pm) = -(1+gamma u) r_pm + (m-u) - [(gamma-1)u+2+h] + 2(1+gamma u) r_pm ...
    即 S(r_pm) = RH(r_pm) + (1+gamma u) r_pm + (m-u) - (gamma-1)u - 2 - h
              = (1+gamma u) r_pm + m + u ... 逐步符号核对，并给出
    M2^2 - 1 在低压缩支的显式符号表达 = [ (m-u) + sqrt(D) ... ]；
A10. 音速轨迹上分支身份量 R = 1+gamma u-(gamma+1)u r* 的分子
     rho(u) = -gamma u^2 + [2 gamma - (gamma+1)(h+2-m)] u + 1；
A11. theta 反解：tau = t(1-r)/(1+r t^2)（由 r 定义反解，几何一致性）；
A12. (theta, M) 平面隐式方程：以 s = tan^2(theta) 消元，
     报告 Res_u( E(u), s*den - num ) 的次数与因式结构；
A13. M->infinity 渐近：u_s = (gamma+1)/(2 gamma) * m + O(1)，
     求出 O(1) 修正项（含 Q 的首个出现阶）。

运行:
  PYTHONIOENCODING=utf-8 python sonic_symbolic_audit.py
"""

import sys
import sympy as sp

g, m, h, u, r, t, tau, s, Q = sp.symbols("gamma m h u r t tau s Q", positive=True)

PASS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        print("!! 审计失败，终止。")
        sys.exit(1)


print("=" * 72)
print("A1: 波后总马赫数恒等式（从原始速度分量直接构造）")
# 无量纲化取 R_gas*T1 = 1，则 a1^2 = gamma，V1^2 = gamma*m。
# sin^2(beta) = u/m, cos^2(beta) = 1 - u/m。
P = 1 + g * u * (1 - r)          # p2/p1，来自法向动量守恒（推导文档 2.3）
T2 = P * r                        # T2/T1（推导文档 2.4）
Vn2sq = g * u * r**2              # (V_n2)^2 = (r V_n1)^2, V_n1^2 = gamma*u
Vt_sq = g * m * (1 - u / m)       # V_t^2 = V1^2 cos^2(beta) = gamma*(m-u)
M2sq_raw = (Vn2sq + Vt_sq) / (g * T2)
M2sq_claim = (u * r**2 + m - u) / (P * r)
check("A1 M2^2 = (u r^2 + m - u)/(P r)", sp.simplify(M2sq_raw - M2sq_claim) == 0)

print("A2: 法向/切向分解 M2^2 = M2n^2 + (m-u)/(P r)")
M2n_sq = u * r / P
check("A2", sp.simplify(M2sq_claim - (M2n_sq + (m - u) / (P * r))) == 0)

print("=" * 72)
print("A3: M2=1 等价于 S(r)=0（物理域 P>0, r>0 下分母非零）")
S = (g + 1) * u * r**2 - (1 + g * u) * r + (m - u)
check("A3", sp.simplify(sp.together(M2sq_claim - 1) * (P * r) - S) == 0,
      "M2^2-1 = S/(P r)")

print("A4: 与 R-H 主方程线性消去 r^2 -> r*")
RH = (g + 1) * u * r**2 - 2 * (1 + g * u) * r + (g - 1) * u + 2 + h
diff_lin = sp.expand(RH - S)  # r^2 项相消
check("A4a RH-S 关于 r 是线性的", sp.degree(sp.Poly(diff_lin, r)) == 1)
r_star = sp.solve(diff_lin, r)[0]
r_star_claim = (g * u + h + 2 - m) / (1 + g * u)
check("A4b r* = (gamma u + h + 2 - m)/(1 + gamma u)",
      sp.simplify(r_star - r_star_claim) == 0)

print("=" * 72)
print("A5: 代回 RH -> u 的二次方程（u^3 系数恒为零）")
E_expr = sp.expand(sp.cancel(RH.subs(r, r_star) * (1 + g * u) ** 2))
E_poly = sp.Poly(E_expr, u)
check("A5a deg_u = 2（三次项相消）", E_poly.degree() == 2,
      f"deg={E_poly.degree()}")
B = h - (m - 1)
c2_claim = g * (g * h + 2 * B)
c1_claim = (g + 1) * B**2 + 2 * B + 2 * g * (m - 1)
c0_claim = 2 * (m - 1) - h
E_claim = c2_claim * u**2 + c1_claim * u + c0_claim
check("A5b 系数 = {c2, c1, c0} 声称式", sp.expand(E_expr - E_claim) == 0)
print("    c2 = gamma*(gamma*h + 2B),  B = h-(m-1)")
print("    c1 = (gamma+1)B^2 + 2B + 2 gamma(m-1)")
print("    c0 = 2(m-1) - h")
print("    注: gamma*h = 2(gamma-1)Q，故 c2 = gamma*[(gamma+2)h - 2(m-1)]")
check("A5c c2 恒等变形", sp.simplify(c2_claim - g * ((g + 2) * h - 2 * (m - 1))) == 0)

print("A6: 与 Res_r(RH,S) 一致（消元路线交叉核对）")
res = sp.resultant(sp.Poly(RH, r), sp.Poly(S, r))
ratio = sp.cancel(res / E_expr)
ratio_poly = sp.Poly(ratio, u)
check("A6 Res_r(RH,S)/E 为 u 的单项式（物理域 u>0 内非零）",
      ratio_poly.is_monomial and sp.simplify(ratio) != 0,
      f"ratio={sp.factor(ratio)}")

print("=" * 72)
print("A7: 正波端点 m=u：E 以 u=u_CJ 为根")
E_normal = sp.expand(E_claim.subs(m, u))
# u_CJ 满足 u^2 - 2*Acal*u + 1 = 0, Acal = 1 + (gamma^2-1)Q/gamma = 1 + (gamma+1)h/2
Acal = 1 + (g + 1) * h / 2
cj_poly = u**2 - 2 * Acal * u + 1
quot, rem = sp.div(sp.Poly(E_normal, u), sp.Poly(cj_poly, u))
check("A7 E|_{m=u} 可被 CJ 二次式整除", sp.simplify(rem.as_expr()) == 0,
      f"商={sp.factor(quot.as_expr())}")

print("=" * 72)
print("A8: 惰性极限 h=0 与独立重建的惰性音速条件一致")
# 惰性 R-H 唯一压缩根: r_inert = (2 + (gamma-1)u) / ((gamma+1)u)
r_inert = (2 + (g - 1) * u) / ((g + 1) * u)
S_inert = sp.together(S.subs(r, r_inert))
S_inert_num = sp.numer(sp.cancel(S_inert))  # 分母 (gamma+1)u > 0
E_h0 = sp.expand(E_claim.subs(h, 0))
ratio8 = sp.cancel(sp.factor(E_h0) / sp.factor(S_inert_num))
check("A8 E|_{h=0} 与 S(r_inert) 分子成常数(含参数 m)比",
      sp.simplify(sp.diff(ratio8, u)) == 0,
      f"ratio={sp.factor(ratio8)}")

print("=" * 72)
print("A9: 低压缩支恒超声速（除 CJ 点）")
# R-H 两根 r_pm = (1+gamma u ± sqrt(D)) / ((gamma+1)u)，D=(u-1)^2-(gamma+1)u h
D_expr = (u - 1) ** 2 - (g + 1) * u * h
sqrtD = sp.symbols("sqrtD", nonnegative=True)
r_L = (1 + g * u + sqrtD) / ((g + 1) * u)
S_at_rL = sp.expand(S.subs(r, r_L))
# 用 sqrtD^2 = D 化简
S_at_rL = sp.expand(S_at_rL.subs(sqrtD**2, D_expr))
S_at_rL = sp.collect(S_at_rL, sqrtD)
# 声称: S(r_L) = [ sqrtD*(sqrtD + (1+gamma u)) ] / ((gamma+1)u) + (m-u) - u + ...
# 直接验证闭式: S(r_L) = ( sqrtD^2 + (1+gamma u) sqrtD )/((gamma+1)u) ... 展开核对
S_claim_L = (D_expr + (1 + g * u) * sqrtD) / ((g + 1) * u) + (m - u)
# 注意 S(r) = (gamma+1)u r^2 - (1+gamma u) r + (m-u)，而 r_L 满足
# (gamma+1)u r_L^2 - (1+gamma u) r_L = r_L*[(gamma+1)u r_L - (1+gamma u)] = r_L*sqrtD
S_claim_L2 = r_L * sqrtD + (m - u)
check("A9a S(r_L) = r_L*sqrt(D) + (m-u) >= 0（逐项非负）",
      sp.simplify(S_at_rL - sp.expand(S_claim_L2.subs(sqrtD**2, D_expr))) == 0)
print("    r_L>0, sqrt(D)>=0, m>=u => S(r_L)>=0，且=0 当且仅当 D=0 且 m=u")
print("    即低压缩支 M2>1，唯一例外是正 CJ 波（M=M_CJ, theta=0）")
r_H_sym = (1 + g * u - sqrtD) / ((g + 1) * u)
S_at_rH = sp.expand(S.subs(r, r_H_sym).subs(sqrtD**2, D_expr))
S_claim_H = sp.expand((-r_H_sym * sqrtD + (m - u)).subs(sqrtD**2, D_expr))
check("A9b S(r_H) = -r_H*sqrt(D) + (m-u)（高压缩支两项竞争）",
      sp.simplify(S_at_rH - S_claim_H) == 0)

print("=" * 72)
print("A10: 音速轨迹上的分支身份量 R")
R_branch = 1 + g * u - (g + 1) * u * r_star_claim
rho_claim = -g * u**2 + (2 * g - (g + 1) * (h + 2 - m)) * u + 1
check("A10 R = rho(u)/(1+gamma u), rho 为开口向下二次式",
      sp.simplify(sp.together(R_branch) - rho_claim / (1 + g * u)) == 0)

print("=" * 72)
print("A11: 由 (t, r) 反解 tau（theta 显式化）")
r_geom = (t - tau) / (t * (1 + t * tau))
tau_solved = sp.solve(sp.Eq(r, r_geom), tau)
tau_claim = t * (1 - r) / (1 + r * t**2)
check("A11 tau = t(1-r)/(1+r t^2)",
      any(sp.simplify(sol - tau_claim) == 0 for sol in tau_solved))

print("=" * 72)
print("A12: (theta,M) 平面隐式方程（报告次数与因式结构）")
# s = tau^2 = t^2 (1-r)^2 / (1+r t^2)^2, t^2 = u/(m-u)
t2 = u / (m - u)
s_expr = sp.cancel(t2 * (1 - r_star_claim) ** 2 / (1 + r_star_claim * t2) ** 2)
s_num, s_den = sp.fraction(s_expr)
G_poly = sp.Poly(sp.expand(s * s_den - s_num), u)
print(f"    s(u) 有理式: 分子/分母关于 u 次数 = "
      f"{sp.degree(sp.Poly(s_num, u))}/{sp.degree(sp.Poly(s_den, u))}")
res_su = sp.resultant(sp.Poly(E_claim, u), G_poly)
res_su_f = sp.factor(res_su)
Ps = sp.Poly(res_su, s)
print(f"    Res_u 关于 s 的次数 = {Ps.degree()}")
print(f"    因式结构: {sp.factor_list(res_su_f)[1].__len__()} 个非平凡因子")
# 打印各因子的 (s次数, m次数)
for fac, mult in sp.factor_list(res_su_f)[1]:
    ds = sp.degree(sp.Poly(fac, s)) if fac.has(s) else 0
    dm = sp.degree(sp.Poly(fac, m)) if fac.has(m) else 0
    print(f"      factor(deg_s={ds}, deg_m={dm}, mult={mult}): "
          f"{sp.sstr(fac)[:120]}{'...' if len(sp.sstr(fac))>120 else ''}")
check("A12 隐式消元完成（次数已报告）", True)

print("=" * 72)
print("A13: M->infinity 渐近")
eps = sp.symbols("epsilon", positive=True)
# u = alpha*m + delta, 领头阶
alpha = sp.Rational(1, 1) * (g + 1) / (2 * g)
delta = sp.symbols("delta")
E_sub = sp.expand(E_claim.subs(u, alpha * m + delta))
E_ser = sp.Poly(E_sub, m)
lead = E_ser.coeff_monomial(m**3)
check("A13a u = (gamma+1)m/(2gamma) 消去 m^3 领头项", sp.simplify(lead) == 0,
      f"m^3 系数 = {sp.simplify(lead)}")
c_m2 = sp.simplify(E_ser.coeff_monomial(m**2))
delta_star = sp.solve(sp.Eq(c_m2, 0), delta)
check("A13b O(m^2) 平衡对 delta 线性可解", len(delta_star) == 1)
delta_simp = sp.simplify(sp.expand(delta_star[0]))
print(f"    delta* = {delta_simp}")
print(f"    因式形: delta* = {sp.factor(delta_simp)}")
# 惰性部分与含 Q 部分拆开（h = 2(gamma-1)Q/gamma）
d_h0 = delta_simp.subs(h, 0)
d_Q = sp.simplify(delta_simp - d_h0)
print(f"    惰性部分 delta*(h=0) = {sp.factor(d_h0)}")
print(f"    放热修正 = {sp.factor(d_Q)}  （h 的一次式：Q 首次出现在 O(1) 阶）")
check("A13c 放热修正对 h 为线性", sp.degree(sp.Poly(d_Q, h)) == 1)

print("=" * 72)
n_ok = sum(1 for _, ok in PASS if ok)
print(f"审计完成: {n_ok}/{len(PASS)} 项全部 PASS")
