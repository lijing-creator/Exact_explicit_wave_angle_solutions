# -*- coding: utf-8 -*-
"""
审计声明："爆轰版 θ–β–M 无根式精确闭式"。

方法：对若干组精确有理参数 (tanθ=τ, M², γ, Q)，把含根号的爆轰 θ–β–M
关系有理化成 tanβ 的多项式 P(t)，然后：
  1) 记录 P 的次数与在 Q 上的不可约因子分解；
  2) 数值求解原方程（过驱动支），定位物理根所在的因子；
  3) 对该因子（次数≤6 时）计算 Galois 群，检查可解性。
若物理根所在因子不可约、次数>4 且 Galois 群不可解，则该实例无根式解
⇒ 一般参数下不存在根式闭式（存在闭式则每个实例都必须可解）。

对照组：Q=0（惰性）应退化为三次因子（Thompson 1950/Mascitti 1969 可解）。

关系式（与 MATLAB solve_postwave_parameters.m 同约定）：
  u = Mn² = M² t²/(1+t²),  t = tanβ
  X = (γ+1)u / (1+γu − √D),  D = (u−1)² − 2(γ²−1)/γ·u·Q     (过驱动支)
  几何: X = tanβ/tan(β−θ) = t(1+tτ)/(t−τ)
有理化: √D = 1+γu − (γ+1)u/X_geo，两边平方后清分母。
"""
import sympy as sp
from sympy import Rational as R

t = sp.symbols('t')

def build_poly(tau, M2, g, Q):
    u = M2 * t**2 / (1 + t**2)
    Xgeo = t * (1 + t * tau) / (t - tau)
    D = (u - 1)**2 - 2 * (g**2 - 1) / g * u * Q
    expr = D - (1 + g * u - (g + 1) * u / Xgeo)**2
    num, _ = sp.fraction(sp.cancel(sp.together(expr)))
    P = sp.Poly(sp.expand(num), t)
    return P

def physical_root(P, tau, M2, g, Q):
    """在 P 的数值根中筛出弱过驱动物理根（验证未平方的原方程）。"""
    import mpmath as mp
    mp.mp.dps = 40
    theta = mp.atan(mp.mpf(tau.p) / mp.mpf(tau.q))
    gq = mp.mpf(g.p) / g.q
    Qq = mp.mpf(Q.p) / Q.q
    M2q = mp.mpf(M2.p) / M2.q
    cands = []
    for r in P.nroots(n=40):
        if abs(sp.im(r)) > 1e-25:
            continue
        tv = mp.mpf(str(sp.re(r)))
        if tv <= mp.tan(theta):        # 需要 β > θ
            continue
        beta = mp.atan(tv)
        u = M2q * mp.sin(beta)**2
        Dv = (u - 1)**2 - 2 * (gq**2 - 1) / gq * u * Qq
        if Dv < 0:
            continue
        X_rh = (gq + 1) * u / (1 + gq * u - mp.sqrt(Dv))   # 过驱动支
        X_geo = mp.tan(beta) / mp.tan(beta - theta)
        if abs(X_rh - X_geo) / X_geo < mp.mpf('1e-20'):
            cands.append((beta, tv))
    if not cands:
        raise RuntimeError('未找到过驱动物理根')
    cands.sort()
    return cands[0][1], cands[0][0]    # 弱过驱动 = 最小 β

def audit(tag, tau, M2, g, Q):
    print(f'\n=== {tag}: tanθ={tau}, M²={M2}, γ={g}, Q={Q} ===')
    P = build_poly(tau, M2, g, Q)
    print(f'P(t) 总次数 = {P.degree()}')
    _, factors = sp.factor_list(P)
    import mpmath as mp
    tstar, bstar = physical_root(P, tau, M2, g, Q)
    print(f'物理根: β* = {mp.nstr(bstar*180/mp.pi, 8)}°, t* = {mp.nstr(tstar, 10)}')
    for i, (fac, mult) in enumerate(factors):
        pf = sp.Poly(fac, t)
        d = pf.degree()
        if d == 0:
            continue
        val = abs(complex(pf.as_expr().evalf(30, subs={t: sp.Float(mp.nstr(tstar, 30), 30)})))
        scale = max(abs(float(c)) for c in pf.all_coeffs())
        hit = val / scale < 1e-15
        line = f'  因子{i}: 次数 {d}, 重数 {mult}, 含物理根: {hit}'
        if hit:
            # factor_list 的因子在 Q 上不可约
            if d <= 6:
                try:
                    G, _alt = sp.galois_group(pf, by_name=False)
                    line += (f' | Galois 群阶 = {G.order()}, 可解 = {G.is_solvable}')
                except Exception as e:
                    line += f' | Galois 群计算失败: {type(e).__name__}: {e}'
            else:
                line += ' | 次数>6（次数>4 的不可约因子已排除通用根式解；逐例可解性需另证）'
        print(line)

if __name__ == '__main__':
    print('sympy', sp.__version__)
    # 对照组：惰性 Q=0，应见三次可解因子
    audit('对照组-惰性', R(1, 2), R(49), R(13, 10), R(0))
    # 实例1：Main.m 工况附近（γ=1.3, M=7, Q=10, θ≈26.57° ∈ [11°, 40.4°]）
    audit('实例1', R(1, 2), R(49), R(13, 10), R(10))
    # 实例2：另一组通用参数（γ=7/5, M=6, Q=5, θ≈18.43°）
    audit('实例2', R(1, 3), R(36), R(7, 5), R(5))
    # 实例3：低放热（γ=6/5, M²=64, Q=2, θ≈14°）
    audit('实例3', R(1, 4), R(64), R(6, 5), R(2))
