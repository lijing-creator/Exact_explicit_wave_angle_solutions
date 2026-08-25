#!/usr/bin/env python3
"""
爆轰 theta-beta-M 精确反演的数值验证程序。

本程序完成三类相互独立的核对：

1. 用 Cardano/三角公式求推导所得的 tan(beta) 三次方程；
2. 用 Brent 法直接求原始含根号的高压缩支方程；
3. 用解析重根三次式求脱体点，并与原始极曲线上的数值极大值比较。

角度在内部全部使用弧度，CSV 和终端摘要中的角度使用度。
热释放参数 Q 的无量纲定义与推导文档完全相同。
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar


RAD_TO_DEG = 180.0 / math.pi
ROOT_TOL = 2.0e-11
FILTER_TOL = 2.0e-9


@dataclass(frozen=True)
class Params:
    """一组上游参数。"""

    M: float
    gamma: float
    Q: float

    @property
    def m(self) -> float:
        return self.M * self.M

    @property
    def qprime(self) -> float:
        return 2.0 * (self.gamma - 1.0) * self.Q / self.gamma


@dataclass(frozen=True)
class CJPoint:
    """CJ 起点。"""

    M_cj: float
    u_cj: float
    beta_cj: float
    theta_cj: float
    X_cj: float


@dataclass(frozen=True)
class DetachmentPoint:
    """附体极曲线的脱体极限。"""

    theta_max: float
    beta_max: float
    s: float
    y: float


def scaled_tolerance(*values: float, factor: float = ROOT_TOL) -> float:
    return factor * max(1.0, *(abs(v) for v in values))


def unique_sorted(values: Iterable[float], tol: float = 2.0e-10) -> list[float]:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    result: list[float] = []
    for value in values:
        if not result or abs(value - result[-1]) > tol * max(1.0, abs(value)):
            result.append(value)
    return result


def real_cubic_roots(
    a: float, b: float, c: float, d: float, tol: float = ROOT_TOL
) -> list[float]:
    """
    用实数 Cardano/三角公式求 a*x^3+b*x^2+c*x+d=0 的全部不同实根。

    这里不调用 numpy.roots，以便被验证的对象确实是推导中的闭式公式。
    仅在首项因浮点舍入退化时，降为二次或一次方程。
    """

    coefficient_scale = max(1.0, abs(a), abs(b), abs(c), abs(d))
    eps = tol * coefficient_scale

    if abs(a) <= eps:
        if abs(b) <= eps:
            if abs(c) <= eps:
                return []
            return [-d / c]
        discriminant = c * c - 4.0 * b * d
        disc_tol = tol * max(1.0, c * c, abs(4.0 * b * d))
        if discriminant < -disc_tol:
            return []
        discriminant = max(0.0, discriminant)
        sqrt_disc = math.sqrt(discriminant)
        return unique_sorted(
            [(-c + sqrt_disc) / (2.0 * b), (-c - sqrt_disc) / (2.0 * b)]
        )

    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (
        27.0 * a * a * d
        - 9.0 * a * b * c
        + 2.0 * b * b * b
    ) / (27.0 * a * a * a)
    shift = b / (3.0 * a)
    discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
    disc_scale = max(1.0, (0.5 * q) ** 2, abs((p / 3.0) ** 3))
    disc_tol = tol * disc_scale

    if discriminant > disc_tol:
        sqrt_disc = math.sqrt(discriminant)
        z = float(np.cbrt(-0.5 * q + sqrt_disc) + np.cbrt(-0.5 * q - sqrt_disc))
        return [z - shift]

    if discriminant >= -disc_tol:
        if abs(q) <= scaled_tolerance(p, q, factor=tol):
            return [-shift]
        u = float(np.cbrt(-0.5 * q))
        return unique_sorted([2.0 * u - shift, -u - shift])

    if p >= 0.0:
        raise ArithmeticError(
            "三实根分支出现 p>=0，说明浮点判别式或系数量级异常。"
        )

    cosine_argument = (
        (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
    )
    cosine_argument = min(1.0, max(-1.0, cosine_argument))
    phi = math.acos(cosine_argument) / 3.0
    amplitude = 2.0 * math.sqrt(-p / 3.0)
    roots = [
        amplitude * math.cos(phi - 2.0 * math.pi * k / 3.0) - shift
        for k in range(3)
    ]
    return unique_sorted(roots)


def cj_point(params: Params) -> CJPoint:
    """由 D=0 求物理 CJ 法向马赫数及其斜波起点。"""

    gamma = params.gamma
    A_cj = 1.0 + (gamma * gamma - 1.0) * params.Q / gamma
    if A_cj < 1.0:
        raise ValueError("当前参数导致 CJ 根号无实数值。")
    u_cj = A_cj + math.sqrt(max(0.0, A_cj * A_cj - 1.0))
    M_cj = math.sqrt(u_cj)
    if params.M <= M_cj:
        raise ValueError(
            f"M={params.M:g} 不大于 M_CJ={M_cj:.12g}，"
            "不存在非退化的附体斜弱过驱动支。"
        )
    beta_cj = math.asin(M_cj / params.M)
    X_cj = (gamma + 1.0) * u_cj / (1.0 + gamma * u_cj)
    theta_cj = beta_cj - math.atan(math.tan(beta_cj) / X_cj)
    return CJPoint(M_cj, u_cj, beta_cj, theta_cj, X_cj)


def normal_mach_squared(beta: float, params: Params) -> float:
    return params.m * math.sin(beta) ** 2


def radical_D_from_u(u: float, params: Params) -> float:
    gamma = params.gamma
    return (
        (u - 1.0) ** 2
        - 2.0 * (gamma * gamma - 1.0) * u * params.Q / gamma
    )


def compression_ratio_high(beta: float, params: Params) -> float:
    """原始 R-H 关系的高压缩根 X_+。"""

    u = normal_mach_squared(beta, params)
    D = radical_D_from_u(u, params)
    d_tol = 5.0e-12 * max(1.0, (u - 1.0) ** 2)
    if D < -d_tol:
        return math.nan
    sqrt_D = math.sqrt(max(0.0, D))
    denominator = 1.0 + params.gamma * u - sqrt_D
    if denominator <= 0.0:
        return math.nan
    return (params.gamma + 1.0) * u / denominator


def theta_from_beta_high(beta: float, params: Params) -> float:
    """在原始含根号高压缩极曲线上，由 beta 直接计算 theta。"""

    X = compression_ratio_high(beta, params)
    if not math.isfinite(X) or X <= 0.0:
        return math.nan
    return beta - math.atan(math.tan(beta) / X)


def original_ratio_residual(beta: float, theta: float, params: Params) -> float:
    """
    原始方程的相对残差：

        tan(beta)/tan(beta-theta) = X_+(beta).
    """

    X = compression_ratio_high(beta, params)
    X_geo = math.tan(beta) / math.tan(beta - theta)
    return abs(X_geo - X) / max(1.0, abs(X))


def cubic_coefficients(theta: float, params: Params) -> tuple[float, ...]:
    tau = math.tan(theta)
    m = params.m
    qp = params.qprime
    a3 = tau * tau * (qp + (params.gamma - 1.0) * m + 2.0)
    a2 = 2.0 * tau * (qp - (m - 1.0))
    a1 = qp + tau * tau * ((params.gamma + 1.0) * m + 2.0)
    a0 = 2.0 * tau
    return a3, a2, a1, a0


def normalized_polynomial_residual(
    x: float, coefficients: Sequence[float]
) -> float:
    value = 0.0
    scale = 0.0
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        value += coefficient * x**power
        scale += abs(coefficient) * abs(x) ** power
    return abs(value) / max(1.0, scale)


def root_diagnostics(
    t: float, theta: float, params: Params, cj: CJPoint
) -> dict[str, float | bool]:
    tau = math.tan(theta)
    u = params.m * t * t / (1.0 + t * t)
    r = (t - tau) / (t * (1.0 + t * tau))
    R = 1.0 + params.gamma * u - (params.gamma + 1.0) * u * r
    D = radical_D_from_u(u, params)
    beta = math.atan(t)
    coefficients = cubic_coefficients(theta, params)
    geometry_ok = t > tau - FILTER_TOL * max(1.0, abs(tau))
    cj_ok = u >= cj.u_cj - FILTER_TOL * max(1.0, cj.u_cj)
    high_compression_ok = R >= -FILTER_TOL * max(1.0, abs(R))
    return {
        "t": t,
        "beta": beta,
        "u": u,
        "r": r,
        "R": R,
        "D": D,
        "D_minus_R2": D - R * R,
        "cubic_residual": normalized_polynomial_residual(t, coefficients),
        "geometry_ok": geometry_ok,
        "cj_ok": cj_ok,
        "high_compression_ok": high_compression_ok,
        "accepted": geometry_ok and cj_ok and high_compression_ok,
    }


def closed_form_weak_beta(
    theta: float, params: Params, cj: CJPoint
) -> tuple[float, list[dict[str, float | bool]]]:
    """
    求三次式的全部实根，按几何、CJ 和平方前符号条件筛选，
    再取高压缩支中较小的 beta。
    """

    # CJ 是根号判别式 D=0 的分支端点。完整分段闭式在此处本来就应使用
    # 独立的 beta_CJ 公式；若仍从三次式取近似根，D 中的相消会把极小的
    # beta 舍入误差放大成 O(sqrt(eps)) 的回代误差。Q=0 时原三次式还会
    # 在 theta=0 进一步退化为 0=0。
    cj_theta_tol = ROOT_TOL * max(1.0, abs(cj.theta_cj))
    if abs(theta - cj.theta_cj) <= cj_theta_tol:
        t_cj = math.tan(cj.beta_cj)
        diagnostics = [root_diagnostics(t_cj, theta, params, cj)]
        return cj.beta_cj, diagnostics

    roots = real_cubic_roots(*cubic_coefficients(theta, params))
    diagnostics = [root_diagnostics(t, theta, params, cj) for t in roots]
    accepted = [item for item in diagnostics if bool(item["accepted"])]
    if not accepted:
        detail = ", ".join(
            f"t={float(item['t']):.8g}, "
            f"u={float(item['u']):.8g}, R={float(item['R']):.8g}"
            for item in diagnostics
        )
        raise RuntimeError(f"没有找到合法的高压缩根。候选根：{detail}")
    selected = min(accepted, key=lambda item: float(item["beta"]))
    return float(selected["beta"]), diagnostics


def closed_form_detachment(params: Params, cj: CJPoint) -> DetachmentPoint:
    """用重根条件导出的 y 三次式求解析脱体点。"""

    m = params.m
    qp = params.qprime
    A = qp + (params.gamma - 1.0) * m + 2.0
    B = qp - (m - 1.0)
    C = (params.gamma + 1.0) * m + 2.0
    y_roots = real_cubic_roots(A * C, B * C + 3.0 * A, 4.0 * B, qp)

    candidates: list[DetachmentPoint] = []
    for y in y_roots:
        if y <= 0.0:
            continue
        s = -(3.0 * A * y * y + 4.0 * B * y + qp) / C
        if s <= 0.0:
            continue
        tau = math.sqrt(s)
        t = y / tau
        if t <= tau:
            continue
        theta = math.atan(tau)
        beta = math.atan(t)
        diagnostics = root_diagnostics(t, theta, params, cj)
        if not bool(diagnostics["accepted"]):
            continue
        candidates.append(DetachmentPoint(theta, beta, s, y))

    if not candidates:
        raise RuntimeError(
            "脱体重根三次式没有通过 y>0、s>0、几何、CJ 和高压缩条件的根。"
        )
    return max(candidates, key=lambda point: point.s)


def numerical_detachment(params: Params, cj: CJPoint) -> DetachmentPoint:
    """
    不使用三次式，直接在原始含根号极曲线上数值最大化 theta(beta)。
    """

    lower = cj.beta_cj
    upper = 0.5 * math.pi - 2.0e-9

    def objective(beta: float) -> float:
        theta = theta_from_beta_high(beta, params)
        if not math.isfinite(theta):
            return 1.0e6
        return -theta

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 800},
    )
    if not result.success:
        raise RuntimeError(f"数值极值搜索失败：{result.message}")
    beta = float(result.x)
    theta = theta_from_beta_high(beta, params)
    if not math.isfinite(theta):
        raise RuntimeError("数值极值落在原始含根号方程的非实区域。")
    tau = math.tan(theta)
    return DetachmentPoint(theta, beta, tau * tau, tau * math.tan(beta))


def numerical_weak_beta(
    theta: float,
    params: Params,
    cj: CJPoint,
    detachment_numeric: DetachmentPoint,
) -> float:
    """
    在 [beta_CJ,beta_max] 上用 Brent 法直接求
    theta_original(beta)-theta_target=0。

    此过程只使用原始含根号高压缩关系，不使用推导所得三次式。
    """

    lower = cj.beta_cj
    upper = detachment_numeric.beta_max

    def residual(beta: float) -> float:
        return theta_from_beta_high(beta, params) - theta

    f_lower = residual(lower)
    f_upper = residual(upper)
    endpoint_tol = 2.0e-11
    if abs(f_lower) <= endpoint_tol:
        return lower
    if abs(f_upper) <= endpoint_tol:
        return upper
    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            "Brent 区间未夹住弱支根："
            f"f(beta_CJ)={f_lower:.6e}, f(beta_max)={f_upper:.6e}。"
        )
    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=5.0e-15,
            rtol=8.0 * np.finfo(float).eps,
            maxiter=300,
        )
    )


def case_label(params: Params) -> str:
    return f"M={params.M:g}, gamma={params.gamma:g}, Q={params.Q:g}"


def scan_case(
    params: Params,
    theta_points: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    cj = cj_point(params)
    det_closed = closed_form_detachment(params, cj)
    det_numeric = numerical_detachment(params, cj)

    if det_closed.theta_max <= cj.theta_cj:
        raise RuntimeError("解析脱体角不大于 CJ 起始转角，物理区间为空。")

    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for index, xi in enumerate(np.linspace(0.0, 1.0, theta_points)):
        theta = cj.theta_cj + float(xi) * (
            det_closed.theta_max - cj.theta_cj
        )
        beta_closed, diagnostics = closed_form_weak_beta(theta, params, cj)
        for root_index, item in enumerate(diagnostics):
            candidate_rows.append(
                {
                    "M": params.M,
                    "gamma": params.gamma,
                    "Q": params.Q,
                    "theta_index": index,
                    "xi": float(xi),
                    "theta_deg": theta * RAD_TO_DEG,
                    "root_index": root_index,
                    "t": float(item["t"]),
                    "beta_deg": float(item["beta"]) * RAD_TO_DEG,
                    "u": float(item["u"]),
                    "u_cj": cj.u_cj,
                    "R": float(item["R"]),
                    "D": float(item["D"]),
                    "D_minus_R2": float(item["D_minus_R2"]),
                    "cubic_rel_residual": float(item["cubic_residual"]),
                    "geometry_ok": bool(item["geometry_ok"]),
                    "cj_ok": bool(item["cj_ok"]),
                    "high_compression_ok": bool(
                        item["high_compression_ok"]
                    ),
                    "accepted": bool(item["accepted"]),
                }
            )

        if index == 0:
            beta_numeric = cj.beta_cj
            comparison_kind = "CJ endpoint"
        elif index == theta_points - 1:
            beta_numeric = det_numeric.beta_max
            comparison_kind = "detachment endpoint"
        else:
            beta_numeric = numerical_weak_beta(
                theta, params, cj, det_numeric
            )
            comparison_kind = "same-theta Brent root"

        selected = min(
            (item for item in diagnostics if bool(item["accepted"])),
            key=lambda item: float(item["beta"]),
        )
        beta_error_deg = abs(beta_closed - beta_numeric) * RAD_TO_DEG
        raw_radical_residual = original_ratio_residual(
            beta_closed, theta, params
        )
        if index == 0:
            # CJ 处 D=0，应以精确端点 X_CJ 回代。raw_radical_residual
            # 仍保留直接计算根号判别式时的相消误差，供数值病态诊断。
            X_geo = math.tan(beta_closed) / math.tan(beta_closed - theta)
            stable_original_residual = abs(X_geo - cj.X_cj) / max(
                1.0, abs(cj.X_cj)
            )
            theta_back = beta_closed - math.atan(
                math.tan(beta_closed) / cj.X_cj
            )
        else:
            stable_original_residual = raw_radical_residual
            theta_back = theta_from_beta_high(beta_closed, params)
        theta_back_error_deg = abs(theta_back - theta) * RAD_TO_DEG

        rows.append(
            {
                "M": params.M,
                "gamma": params.gamma,
                "Q": params.Q,
                "index": index,
                "xi": float(xi),
                "comparison_kind": comparison_kind,
                "theta_deg": theta * RAD_TO_DEG,
                "beta_closed_deg": beta_closed * RAD_TO_DEG,
                "beta_numeric_deg": beta_numeric * RAD_TO_DEG,
                "beta_abs_error_deg": beta_error_deg,
                "theta_back_error_deg": theta_back_error_deg,
                "original_ratio_rel_residual": stable_original_residual,
                "raw_radical_rel_residual": raw_radical_residual,
                "cubic_rel_residual": float(selected["cubic_residual"]),
                "u": float(selected["u"]),
                "u_minus_u_cj": float(selected["u"]) - cj.u_cj,
                "R": float(selected["R"]),
                "D": float(selected["D"]),
                "D_minus_R2": float(selected["D_minus_R2"]),
                "real_root_count": len(diagnostics),
                "accepted_root_count": sum(
                    bool(item["accepted"]) for item in diagnostics
                ),
            }
        )

    frame = pd.DataFrame(rows)
    interior = frame[frame["comparison_kind"] == "same-theta Brent root"]
    if interior.empty:
        interior = frame

    summary: dict[str, object] = {
        **asdict(params),
        "status": "OK",
        "message": "",
        "M_cj": cj.M_cj,
        "u_cj": cj.u_cj,
        "theta_cj_deg": cj.theta_cj * RAD_TO_DEG,
        "beta_cj_deg": cj.beta_cj * RAD_TO_DEG,
        "theta_max_closed_deg": det_closed.theta_max * RAD_TO_DEG,
        "theta_max_numeric_deg": det_numeric.theta_max * RAD_TO_DEG,
        "theta_max_abs_error_deg": abs(
            det_closed.theta_max - det_numeric.theta_max
        )
        * RAD_TO_DEG,
        "beta_at_max_closed_deg": det_closed.beta_max * RAD_TO_DEG,
        "beta_at_max_numeric_deg": det_numeric.beta_max * RAD_TO_DEG,
        "beta_at_max_abs_error_deg": abs(
            det_closed.beta_max - det_numeric.beta_max
        )
        * RAD_TO_DEG,
        "max_interior_beta_error_deg": float(
            interior["beta_abs_error_deg"].max()
        ),
        "max_theta_back_error_deg": float(
            frame["theta_back_error_deg"].max()
        ),
        "max_original_ratio_rel_residual": float(
            frame["original_ratio_rel_residual"].max()
        ),
        "max_raw_radical_rel_residual": float(
            frame["raw_radical_rel_residual"].max()
        ),
        "max_cubic_rel_residual": float(
            frame["cubic_rel_residual"].max()
        ),
        "max_abs_D_minus_R2": float(
            frame["D_minus_R2"].abs().max()
        ),
        "theta_point_count": theta_points,
    }
    return rows, candidate_rows, summary


def make_error_plot(scan_df: pd.DataFrame, output_path: Path) -> None:
    valid = scan_df[scan_df["comparison_kind"] == "same-theta Brent root"].copy()
    if valid.empty:
        return
    labels = (
        valid["M"].map(lambda x: f"{x:g}")
        + "/"
        + valid["gamma"].map(lambda x: f"{x:g}")
        + "/"
        + valid["Q"].map(lambda x: f"{x:g}")
    )
    valid["case"] = labels
    cases = list(dict.fromkeys(valid["case"].tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for case in cases:
        part = valid[valid["case"] == case]
        axes[0].semilogy(
            part["xi"],
            np.maximum(part["beta_abs_error_deg"], 1.0e-18),
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            alpha=0.75,
        )
        axes[1].semilogy(
            part["xi"],
            np.maximum(part["original_ratio_rel_residual"], 1.0e-18),
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            alpha=0.75,
        )

    axes[0].set_title("Closed-form beta vs. Brent root")
    axes[0].set_xlabel("normalized position xi")
    axes[0].set_ylabel("absolute beta error (deg)")
    axes[1].set_title("Residual in the original radical equation")
    axes[1].set_xlabel("normalized position xi")
    axes[1].set_ylabel("relative ratio residual")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        f"Detonation theta-beta-M validation, {len(cases)} parameter cases"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def dataframe_to_markdown(
    frame: pd.DataFrame, columns: Sequence[str]
) -> str:
    """无额外 tabulate 依赖地生成紧凑 Markdown 表格。"""

    def format_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6e}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for values in frame[list(columns)].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(format_cell(value) for value in values) + " |")
    return "\n".join(rows)


def write_markdown_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    passed: bool,
    thresholds: dict[str, float],
) -> None:
    ok = summary_df[summary_df["status"] == "OK"].copy()
    lines = [
        "# 爆轰 theta-beta-M 闭式反演数值验证摘要",
        "",
        f"总判定：{'PASS' if passed else 'FAIL'}",
        "",
        "## 判定阈值",
        "",
        f"- 内点波角绝对误差：{thresholds['beta_error_deg']:.3e} deg",
        f"- 解析/数值最大转角误差：{thresholds['theta_max_error_deg']:.3e} deg",
        f"- 原始含根号方程相对残差：{thresholds['original_residual']:.3e}",
        "",
        "## 方法",
        "",
        "1. 闭式侧：对推导得到的 tan(beta) 三次式使用实数 "
        "Cardano/三角公式。",
        "2. 数值侧：在 CJ 到脱体波角之间，直接对原始含根号高压缩方程 "
        "使用 Brent 法。",
        "3. 脱体点：解析侧解重根 y 三次式；数值侧直接最大化原始 "
        "theta(beta) 极曲线。",
        "4. 每个闭式根均重新检查 t>tan(theta)、u>=u_CJ、R>=0 "
        "以及 D-R^2。",
        "",
        "## 分参数结果",
        "",
    ]
    columns = [
        "M",
        "gamma",
        "Q",
        "status",
        "theta_cj_deg",
        "theta_max_closed_deg",
        "theta_max_abs_error_deg",
        "max_interior_beta_error_deg",
        "max_original_ratio_rel_residual",
    ]
    available = [column for column in columns if column in summary_df.columns]
    lines.append(dataframe_to_markdown(summary_df, available))

    lines.extend(["", "## 全局最大误差", ""])
    if not ok.empty:
        lines.extend(
            [
                "- 内点 beta 闭式/Brent 最大绝对误差："
                f"{ok['max_interior_beta_error_deg'].max():.6e} deg",
                "- 解析/数值 theta_max 最大绝对误差："
                f"{ok['theta_max_abs_error_deg'].max():.6e} deg",
                "- 原始含根号方程最大相对残差："
                f"{ok['max_original_ratio_rel_residual'].max():.6e}",
                "- 含 CJ 相消效应的直接根号计算最大残差："
                f"{ok['max_raw_radical_rel_residual'].max():.6e}",
                "- 三次多项式最大相对残差："
                f"{ok['max_cubic_rel_residual'].max():.6e}",
                "- 平方前符号核对 max|D-R^2|："
                f"{ok['max_abs_D_minus_R2'].max():.6e}",
            ]
        )
    else:
        lines.append("- 没有成功完成的参数组。")

    failed_cases = summary_df[summary_df["status"] != "OK"]
    if not failed_cases.empty:
        lines.extend(["", "## 未完成参数组", ""])
        for _, row in failed_cases.iterrows():
            lines.append(
                f"- M={row['M']:g}, gamma={row['gamma']:g}, Q={row['Q']:g}: "
                f"{row['message']}"
            )

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- scan_results.csv：每个网格点的完整结果和筛选诊断量。",
            "- root_candidates.csv：每个三次实根及其逐项物理筛选结果。",
            "- case_summary.csv：每组 M、gamma、Q 的误差汇总。",
            "- validation_error.png：内点波角误差及原方程残差图。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "比较爆轰 theta-beta-M 三次闭式反演与原始含根号方程的数值解。"
        )
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=float,
        default=[5.0, 7.0, 10.0],
        help="上游马赫数网格，默认：5 7 10",
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=[1.2, 1.3, 1.4],
        help="比热比网格，默认：1.2 1.3 1.4",
    )
    parser.add_argument(
        "--Q",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0],
        help="无量纲热释放网格，默认：2 5 10",
    )
    parser.add_argument(
        "--theta-points",
        type=int,
        default=21,
        help="每个参数组从 CJ 到脱体点的 theta 网格数，默认：21",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="输出目录，默认：当前目录下 results",
    )
    parser.add_argument(
        "--beta-error-deg",
        type=float,
        default=1.0e-6,
        help="内点 beta 闭式/数值最大允许误差，单位 deg",
    )
    parser.add_argument(
        "--theta-max-error-deg",
        type=float,
        default=1.0e-7,
        help="解析/数值 theta_max 最大允许误差，单位 deg",
    )
    parser.add_argument(
        "--original-residual",
        type=float,
        default=1.0e-9,
        help="原始含根号方程最大允许相对残差",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.theta_points < 3:
        raise ValueError("--theta-points 至少为 3。")
    if any(M <= 1.0 for M in args.M):
        raise ValueError("所有 M 必须大于 1。")
    if any(gamma <= 1.0 for gamma in args.gamma):
        raise ValueError("所有 gamma 必须大于 1。")
    if any(Q < 0.0 for Q in args.Q):
        raise ValueError("所有 Q 必须非负。")
    if (
        args.beta_error_deg <= 0.0
        or args.theta_max_error_deg <= 0.0
        or args.original_residual <= 0.0
    ):
        raise ValueError("三个误差阈值必须为正数。")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_arguments(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_candidate_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    parameter_grid = itertools.product(args.M, args.gamma, args.Q)

    for M, gamma, Q in parameter_grid:
        params = Params(float(M), float(gamma), float(Q))
        print(f"[scan] {case_label(params)}")
        try:
            rows, candidate_rows, summary = scan_case(
                params, args.theta_points
            )
            all_rows.extend(rows)
            all_candidate_rows.extend(candidate_rows)
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    **asdict(params),
                    "status": "ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                    "theta_point_count": args.theta_points,
                }
            )
            print(f"  [error] {type(exc).__name__}: {exc}", file=sys.stderr)

    scan_df = pd.DataFrame(all_rows)
    candidates_df = pd.DataFrame(all_candidate_rows)
    summary_df = pd.DataFrame(summaries)
    scan_path = output_dir / "scan_results.csv"
    candidates_path = output_dir / "root_candidates.csv"
    summary_path = output_dir / "case_summary.csv"
    markdown_path = output_dir / "validation_summary.md"
    plot_path = output_dir / "validation_error.png"

    scan_df.to_csv(scan_path, index=False, encoding="utf-8-sig")
    candidates_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not scan_df.empty:
        make_error_plot(scan_df, plot_path)

    ok = summary_df[summary_df["status"] == "OK"]
    no_case_errors = len(ok) == len(summary_df) and not summary_df.empty
    thresholds = {
        "beta_error_deg": args.beta_error_deg,
        "theta_max_error_deg": args.theta_max_error_deg,
        "original_residual": args.original_residual,
    }
    within_thresholds = (
        no_case_errors
        and float(ok["max_interior_beta_error_deg"].max())
        <= args.beta_error_deg
        and float(ok["theta_max_abs_error_deg"].max())
        <= args.theta_max_error_deg
        and float(ok["max_original_ratio_rel_residual"].max())
        <= args.original_residual
    )
    passed = bool(within_thresholds)
    write_markdown_summary(
        summary_df, markdown_path, passed=passed, thresholds=thresholds
    )

    print("")
    print(f"判定：{'PASS' if passed else 'FAIL'}")
    if not ok.empty:
        print(
            "内点 beta 最大误差："
            f"{ok['max_interior_beta_error_deg'].max():.6e} deg"
        )
        print(
            "theta_max 最大误差："
            f"{ok['theta_max_abs_error_deg'].max():.6e} deg"
        )
        print(
            "原始方程最大相对残差："
            f"{ok['max_original_ratio_rel_residual'].max():.6e}"
        )
    print(f"详细结果：{markdown_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
