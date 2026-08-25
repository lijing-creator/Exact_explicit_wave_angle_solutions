# -*- coding: utf-8 -*-
"""
符号级验证：爆轰版 θ–β–M 有理化多项式 P(t) 是否对一般参数
(τ=tanθ, m=M², γ, Q) 恒分解为 线性 × 三次。
若是，给出三次式系数 ⇒ 爆轰版存在 Cardano 精确闭式（文献未见）。
"""
import sympy as sp

t, tau, m, g, Q = sp.symbols('t tau m gamma Q', positive=True)

u = m * t**2 / (1 + t**2)
Xgeo = t * (1 + t * tau) / (t - tau)
D = (u - 1)**2 - 2 * (g**2 - 1) / g * u * Q
expr = D - (1 + g * u - (g + 1) * u / Xgeo)**2
num, den = sp.fraction(sp.cancel(sp.together(expr)))
P = sp.Poly(sp.expand(num), t)
print('P(t) 次数 =', P.degree())

F = sp.factor(num)
print('\n因式分解：')
sp.pprint(F, wrap_line=False)

# 提取各因子并整理三次因子系数
print('\n--- 因子明细 ---')
for fac, mult in sp.factor_list(num)[1]:
    pf = sp.Poly(fac, t)
    print(f'\n次数 {pf.degree()}（重数 {mult}）:')
    if pf.degree() >= 1:
        for k, c in enumerate(pf.all_coeffs()):
            print(f'  t^{pf.degree()-k}: {sp.simplify(c)}')

# 验证 Q->0 退化：三次因子应回到惰性 θ-β-M 三次式
print('\n--- Q->0 检查 ---')
for fac, mult in sp.factor_list(num)[1]:
    pf = sp.Poly(fac, t)
    if pf.degree() == 3:
        pf0 = sp.Poly(sp.simplify(fac.subs(Q, 0)), t)
        print('三次因子在 Q=0 时:')
        for k, c in enumerate(pf0.all_coeffs()):
            print(f'  t^{pf0.degree()-k}: {sp.simplify(sp.factor(c))}')
