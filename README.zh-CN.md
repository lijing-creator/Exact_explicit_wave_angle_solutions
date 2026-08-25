[English](README.md) | **简体中文**

# 平衡斜爆轰极曲线波角的精确显式解

本仓库是下述论文的配套代码：

> J. Li and C. Luo, *Exact explicit wave-angle solutions of the equilibrium
> oblique-detonation polar*（投稿中）。

平衡斜爆轰极曲线把楔角 $\theta$、波角 $\beta$、来流马赫数 $M$ 和放热量 $Q$
联系起来。正向走——给定 $\beta$ 求 $\theta$——是初等的。反向走——给定楔角
$\theta$ 求波角——过去通常靠读极曲线图或者数值迭代。

论文证明这个反问题有精确闭式解，本仓库是它的参考实现。**如果你只想由楔角
求波角，一次调用就够。**

```python
import math
from odw_polar import PolarModel

polar = PolarModel(M=7.0, gamma=1.3, Q=10.0)      # Q = Q*/(R T1)
beta = polar.weak_wave_angle(math.radians(30.0))  # 输入楔角
print(math.degrees(beta))                         # -> 45.777017  (度)
```

不需要迭代，不需要初值，不需要求根区间。全部归结为一个三次方程（Cardano
公式）或一个二次方程（一次开方）。

## 模型与约定

量热完全气体，波前波后同一气体常数 $R$ 与同一比热比 $\gamma>1$。放热瞬时完全，
所以波面是附着在直楔上的零厚度平衡间断。

无量纲放热量沿用论文的约定：

```
Q  = Q* / (R T1)                q' = 2 (gamma - 1) Q / gamma
```

其中 `Q*` 是单位质量的化学放热，`T1` 是来流静温。**请核对这个约定与你自己
用的是否一致**——放热量的无量纲化方式有好几种在流通，彼此不能互换。

接口中的角度一律是弧度。需要度数时在调用处用 `math.degrees`。

## 实现了什么

[odw_polar.py](odw_polar.py) 是一个自足的模块，**只依赖 Python 标准库**。
每个接口对应论文中一个带编号的结果。

| 调用 | 返回 | 论文 |
|---|---|---|
| `PolarModel(M, gamma, Q)` | 一组来流参数对应的模型 | §2.1 |
| `.cj` | CJ 端点：`M_cj`、`u_cj`、`beta`、`theta`、`r_cj` | (2.7)、(2.8) |
| `.density_ratios(beta)` | 给定波角处的两个密度比根 `(r_H, r_L)` | (2.6) |
| `.deflection(beta, high_compression)` | 正问题，两支任选 | (2.4) |
| `.cubic_coefficients(theta)` | 波角三次式的系数 `(a3, a2, a1, a0)` | (2.12)、(2.13) |
| `.wave_angle_roots(theta)` | **全部**实根，每个根带筛选标志 | (2.12)、(2.16)、(2.17) |
| `.weak_wave_angle(theta)` | 弱过驱解 | §2.3 |
| `.strong_wave_angle(theta)` | 强解 | §2.3 |
| `.detachment()` | 由重根三次式给出 `theta_max`、`beta_d` | (2.27)–(2.29) |
| `.sonic_coefficients()` | 音速二次式的系数 `(c2, c1, c0)` | (3.5)、(3.6) |
| `.sonic_point()` | 波后总音速点，`M2 = 1` | (3.7)、(3.8) |
| `real_cubic_roots(a,b,c,d)` | 实 Cardano／三角形式的三次求根 | (2.18)–(2.23) |

### 分支筛选

消去密度比得到三次式的同时，也丢掉了每个根的分支身份，所以三次式的一个根
本身并不描述一个物理爆轰解。`wave_angle_roots` 返回全部实根，并附带论文中
那个有序筛选的三个标志：

1. `attached` —— 附着条件 `tan(beta) > tan(theta)`；
2. `above_cj` —— CJ 界 `u = M^2 sin^2(beta) >= u_CJ`；
3. `high_compression` —— 分支量 `sigma = 1 + gamma*u - (gamma+1)*u*r` 的符号，
   高压缩支上为正，低压缩支上为负，见 (2.17)。

**判别式 `D` 做不到第 3 步**：把任一支的分支关系两边平方都得到
`D = sigma^2`，所以 `D >= 0` 在两支上都成立，不含任何符号信息。如果你要自己
重写这套公式，这一步是最容易做错的地方。

通过筛选的根里，波角较小的是弱过驱解，较大的是强解。它们恰存在于
`theta_CJ < theta < theta_max`；超出这个区间时 `weak_wave_angle` 抛
`ValueError`，而不是返回一个会误导人的数。

## 目录结构

```
odw_polar.py            闭式实现（仅标准库）
example.py              示例：运行 `python example.py`
requirements.txt        验证与绘图脚本的依赖
verification/           对每一条闭式关系的独立核对
figures/                论文各图的生成脚本
```

## 验证

闭式关系的对照对象是完全不用这些公式的独立数值解：直接在保留根号的原始极曲线
上做求根和极大化。

```bash
python -m pip install -r requirements.txt
cd verification
python validate_closed_form.py          # PASS 返回 0，FAIL 返回 1
```

在 `M ∈ {5, 7, 10}`、`gamma ∈ {1.2, 1.3, 1.4}`、`Q ∈ {2, 5, 10}` 共 27 组参数上：

| 量 | 闭式 | 独立对照 | 一致到 |
|---|---|---|---|
| 波角 | 三次式 (2.12)，Cardano 求根 | 原始极曲线 (2.3) 的 Brent 根 | `1e-11` 度 |
| 脱体角 | 三次式 (2.28)–(2.29) | 极曲线的有界数值极大化 | `1e-13` 度 |
| 音速角 | 二次式 (3.7)–(3.8) | 沿极曲线求 `M2 = 1` 的 Brent 根 | `1e-13` 度 |

预先生成的结果放在 `verification/results/`，其中 `validation_summary.md`
是可直接阅读的 PASS/FAIL 报告。

### 各脚本分别验什么

| 脚本 | 验证内容 |
|---|---|
| `validate_closed_form.py` | 波角与脱体点对 Brent 求根和直接极大化；分支筛选逐根落实；输出 CSV 与摘要 |
| `symbolic_factorization.py` | 精确因式分解 `N(t) = (1+t^2) C3(t)`、系数 (2.13)，以及 `Q -> 0` 时逐项退化为经典惰性斜激波三次式 |
| `irreducibility_check.py` | 波角三次式的 Galois 群；确认一般情形确实不可约，因此三角形式是必需的 |
| `sonic_symbolic_audit.py` | `M2 = 1` 化为二次式 (3.5)：`u^3` 项系数恒为零，且消元路线与结式一致 |
| `sonic_gap_asymptotics.py` | 音速点的存在唯一性、脱体点严格亚声速，以及给出 `K(gamma)/M^4` 的高马赫展开 |
| `sonic_numerical_check.py` | 音速二次式对独立的 `M2 = 1` Brent 根；分支归属与排序 `theta_CJ < theta_s < theta_max` 的参数扫描 |

任一断言不通过，脚本以非零码退出。

## 图

`figures/` 下是论文各图的生成脚本，输出写在脚本同目录。

| 脚本 | 对应图 |
|---|---|
| `oblique_detonation_geometry_clean.py` | 图 1 —— 几何与记号 |
| `polar_cubic_roots.py` | 图 2 —— 三个实根 |
| `polar_cubic_double_root.py` | 图 3 —— 脱体处的重根 |
| `polar_cubic_positive_discriminant.py` | 图 4 —— 脱体之外只剩一个实根 |
| `critical_structure_map.py` | 图 5 —— `(M, theta)` 平面上的解域 |
| `global_limit_collapses.py` | 图 6 —— 高马赫塌缩 |
| `sonic_detachment_gap_conditions.py` | 图 7 —— 带宽随参数的变化 |

`limiting_structure.py` 与 `limiting_structure_geometric.py` 是后两个脚本共用
的辅助模块。

论文中图 1 的几何示意是在
`oblique_detonation_geometry_clean.py` 的输出上手工修饰而成；脚本复现的是布局
和角度标注，不是最终的排版效果。

## 跑一遍示例

```bash
python example.py
```

对 `M = 7`、`gamma = 1.3`、`Q = 10`，它会打印 CJ 点、音速点、脱体点，30 度楔角
下的两个波角，全部代数根及其筛选标志，以及沿弱支的一次扫描。它报出的亚声速带
宽（`0.002258` 度）就是论文图 5 内嵌图里的那个数。

## 引用

```bibtex
@article{li_luo_odw_polar,
  author  = {Li, Jing and Luo, Changtong},
  title   = {Exact explicit wave-angle solutions of the equilibrium
             oblique-detonation polar},
  note    = {submitted}
}
```

Chapman–Jouguet 端点闭式 (2.7)–(2.8) 不是本文首创：它与 Pratt, Humphrey &
Glenn, *J. Propulsion and Power* **7** (1991) 837–845 的式 (28) 和 (32) 逐位
相同，应当引到那里。本文的贡献是反演三次式、分支筛选判据、固定 `Q` 的脱体
三次式，以及音速二次式。

## 适用范围

这些关系描述的是平衡末态和极曲线的结构。它们不表示有限厚度的诱导区与反应区、
弯曲的过渡结构，也不表示多维与胞格不稳定性。在本模型内，`theta_CJ` 是弱高压缩
支的偏转角下限，**不是**有限速率爆轰的起爆阈值。
