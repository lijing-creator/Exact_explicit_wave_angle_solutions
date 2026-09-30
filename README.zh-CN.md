[English](README.md) | **简体中文**

# 平衡斜爆轰波角的精确显式解

本仓库是下述论文的配套代码：

> J. Li and C. Luo, *Exact explicit wave-angle solutions for equilibrium
> oblique detonations*（投稿中）。

平衡斜爆轰极曲线把楔角 $\theta$、波角 $\beta$、来流马赫数 $M$ 和放热量 $Q$
联系起来。正向走——给定 $\beta$ 求 $\theta$——是初等的。反向走——给定楔角
$\theta$ 求波角——过去通常靠读极曲线图或者数值迭代。

论文证明这个反问题有精确闭式解，本仓库是它的参考实现。仓库还包含第 5 节的化学
平衡求解器，其中闭式解取代了内层的波角迭代。**如果你只想由楔角
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

同一套闭式解也适用于两γ模型，即波前两侧比热比不同的情形：一次精确的参数代换
把它化为单γ模型（论文第 4 节）。见下文[两γ模型](#两γ模型)。

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
| `two_gamma_parameters(M, gamma1, gamma2, Q)` | 两γ模型的等效参数 `(gamma_e, M_e, Q_e)` | (4.9) |
| `two_gamma_polar(M, gamma1, gamma2, Q)` | 等效单γ问题的 `PolarModel` | (4.10)、§4.3 |

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

### 两γ模型

两γ模型波前取 `gamma1`、波后取 `gamma2`，来流马赫数
`M = U1 / sqrt(gamma1 p1 / rho1)`，放热 `Q = Q* / (R1 T1)`。把两侧焓系数之差
并入放热量，就得到单γ问题：

```
gamma_e = gamma2
M_e     = M * sqrt(gamma1 / gamma2)
Q_e     = Q + gamma1/(gamma1 - 1) - gamma2/(gamma2 - 1)
```

这个代换是精确的，压力、密度、速度都不变，所以等效模型的波角、偏转角和各临界点
原样就是两γ模型的结果：

```python
import math
from odw_polar import two_gamma_polar

polar = two_gamma_polar(M=8.0, gamma1=1.4, gamma2=1.2, Q=20.0)
print(math.degrees(polar.detachment().theta_max))                # -> 47.316588
print(math.degrees(polar.weak_wave_angle(math.radians(30.0))))   # -> 41.112323
```

波后状态和 `gamma2` 都不变，所以波后马赫数也不变，`sonic_point()` 不需要换算。
以来流声速为尺度的马赫数则是等效值：`polar.M` 是 `M_e`，`polar.cj.M_cj` 是等效
CJ 马赫数，用 `M = M_e * sqrt(gamma2 / gamma1)` 换回实际值。分支判别要求
`Q_e >= 0`；`Q_e < 0` 时等效问题没有 CJ 点，`two_gamma_polar` 会拒绝。

## 化学平衡应用

`equilibrium_odw/` 实现论文第 5 节：波后产物处于化学平衡、组分物性随温度变化的
斜爆轰。两个求解器都用 Zhang et al. (2022) 的两步迭代：先在固定的波后组分下算流动
状态，再在所得温度和压力下算平衡组分，然后松弛更新组分，直到两者都收敛。两者只在
流动这一步不同。

| 调用 | 流动这一步 |
|---|---|
| `solve_closed_form(...)` | 在当前状态线性化混合物焓，由所得单γ模型的闭式三次式直接给出波角，式 (5.1)–(5.3) |
| `solve_newton(...)` | 对固定组分下的能量方程做 Newton 迭代；每轮外迭代从上一轮的波角起步 |

`solve_newton` 是论文中的对照求解器，是按该方法公开发表的描述独立编写的实现，
不是原作者的程序。

闭式迭代从冻结的来流状态起步（`T2 = T1`、`X2 = X1`），这时线性化模型就是来流气体
的惰性斜激波。接近脱体时，给定偏转角可能超过这个模型的最大偏转角，第一次求三次式
无根。这时把线性化温度移到该模型脱体点的温度（由 (2.28)–(2.29) 闭式给出），再在
给定偏转角下重新迭代。组分保持冻结，这一步不做平衡计算。在 5.2 节的主测试集中，
300 个分支态里有 8 个触发了这一步。

```python
import math
from equilibrium_odw import fuel_air, frozen_sound_speed, solve_closed_form

mix, eq, X1 = fuel_air("H2", "path/to/thermo.inp")     # 化学恰当比 H2–空气
u1 = 8.0 * frozen_sound_speed(mix, X1, 300.0)           # T1 = 300 K 时 M = 8
sol = solve_closed_form(mix, eq, X1, 300.0, 101325.0, u1,
                        math.radians(30.0), branch="weak")
print(math.degrees(sol.beta), sol.T2, sol.p2)
```

`python example_equilibrium.py path/to/thermo.inp` 对这一算例用两个求解器算
强弱两支。

### 热力学数据

本仓库不附带任何热力学数据。请传入 McBride, Zehe & Gordon (NASA TP-2002-211556)
NASA-9 格式的物性库路径，例如 NASA CEA 的 `thermo.inp`，或 Combustion Toolbox 的
`databases/thermo_CT.inp`。`fuel_air` 需要 H2、H、O2、O、OH、HO2、H2O2、H2O、N、
N2、NO，烃类混合物另需燃料本身以及 CO、CO2、CH3、HCO。标准态压力取 1 bar。

5.2 节的结果是用本代码和 Combustion Toolbox 1.2.9 的数据算的，物性通过从
Combustion Toolbox 导出的插值表求值。换用其他数据，包括同一物性库的 NASA-9 多项式，
得到的状态和计时会略有不同。例如直接用 `thermo_CT.inp` 的多项式时，Newton 对照
求解器在一个近脱体的弱支算例（甲烷–空气，`M = 7`，`f = 0.95`）上达到内层步数
上限；闭式求解器 300 个分支态全部收敛。

### 复现 5.2 节

```bash
cd equilibrium_benchmark
python run_main_test_set.py path/to/thermo.inp       # 需要几分钟
```

`main_test_set_tasks.csv` 列出主测试集的 150 个算例：混合物、`M`、来流速度 `U`、
`f` 和给定偏转角。脚本用两个求解器对每个算例求强弱两支，报告收敛情况和两个求解器
之间的一致程度，并输出计时表（表 2）：先预热一轮，再取七轮的中位数。

## 目录结构

```
odw_polar.py            闭式实现（仅标准库）
example.py              示例：运行 `python example.py`
equilibrium_odw/        第 5 节的化学平衡求解器（需 numpy）
example_equilibrium.py  第 5 节的示例（需 NASA-9 物性库）
equilibrium_benchmark/  5.2 节的主测试集
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

`two_gamma_check.py` 核对两γ代换：对照对象是直接数值求解两γ间断关系
(4.1)–(4.3)，完全不用等效参数。在 43 组参数上（含 `gamma1 < gamma2` 与
`gamma1 = gamma2`），CJ 点、脱体点、音速点和两支波角都一致到 `1e-12` 度。

预先生成的结果放在 `verification/results/`，其中 `validation_summary.md`
是可直接阅读的 PASS/FAIL 报告。

### 各脚本分别验什么

| 脚本 | 验证内容 |
|---|---|
| `validate_closed_form.py` | 波角与脱体点对 Brent 求根和直接极大化；分支筛选逐根落实；输出 CSV 与摘要 |
| `symbolic_factorization.py` | 精确因式分解 `N(t) = (1+t^2) C3(t)`、系数 (2.13)，以及 `Q -> 0` 时逐项退化为经典惰性斜激波三次式 |
| `irreducibility_check.py` | 波角三次式的 Galois 群；确认一般情形确实不可约，因此三角形式是必需的 |
| `sonic_symbolic_audit.py` | `M2 = 1` 化为二次式 (3.5)：`u^3` 项系数恒为零，且消元路线与结式一致 |
| `sonic_structure_audit.py` | 音速点的存在唯一性，以及脱体点波后严格亚声速 |
| `sonic_numerical_check.py` | 音速二次式对独立的 `M2 = 1` Brent 根；分支归属与排序 `theta_CJ < theta_s < theta_max` 的参数扫描 |
| `two_gamma_check.py` | 两γ代换 (4.9)–(4.10) 对直接求解两γ间断关系 |

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

论文中图 1 的几何示意是在
`oblique_detonation_geometry_clean.py` 的输出上手工修饰而成；脚本复现的是布局
和角度标注，不是最终的排版效果。

## 跑一遍示例

```bash
python example.py
```

对 `M = 7`、`gamma = 1.3`、`Q = 10`，它会打印 CJ 点、音速点、脱体点，30 度楔角
下的两个波角，全部代数根及其筛选标志，以及沿弱支的一次扫描。它报出的弱支亚声速
段宽度（`0.002258` 度）就是论文图 5 内嵌图里的那个数。

## 引用

```bibtex
@article{li_luo_odw_polar,
  author  = {Li, Jing and Luo, Changtong},
  title   = {Exact explicit wave-angle solutions for equilibrium
             oblique detonations},
  note    = {submitted},
  url     = {https://github.com/lijing-creator/Exact_explicit_wave_angle_solutions}
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
