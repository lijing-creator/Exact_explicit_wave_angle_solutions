#!/usr/bin/env python3
"""Minimal worked example for the closed-form oblique-detonation polar.

Run:  python example.py
"""

import math

from odw_polar import PolarModel


def deg(x: float) -> float:
    return math.degrees(x)


def main() -> None:
    # Upstream state: M = 7, gamma = 1.3, Q = Q*/(R T1) = 10.
    polar = PolarModel(M=7.0, gamma=1.3, Q=10.0)
    print(polar)

    # --- Endpoints of the attached polar, all in closed form ---------------
    cj = polar.cj
    det = polar.detachment()
    son = polar.sonic_point()

    print("\ncharacteristic points")
    print(f"  M_CJ                     {cj.M_cj:12.6f}")
    print(f"  CJ            beta,theta {deg(cj.beta):9.4f} {deg(cj.theta):9.4f} deg")
    print(f"  total-sonic   beta,theta {deg(son.beta):9.4f} {deg(son.theta):9.4f} deg")
    print(f"  detachment    beta,theta {deg(det.beta):9.4f} {deg(det.theta_max):9.4f} deg")
    print(f"  strict ordering theta_CJ < theta_s < theta_max: "
          f"{cj.theta < son.theta < det.theta_max}")
    print(f"  subsonic strip width     {deg(det.theta_max - son.theta):12.6f} deg")

    # --- The inverse problem: wedge angle in, wave angles out -------------
    theta = math.radians(30.0)
    print(f"\ninverse problem at theta = {deg(theta):.1f} deg")
    print(f"  weak (overdriven)  beta  {deg(polar.weak_wave_angle(theta)):9.4f} deg")
    print(f"  strong             beta  {deg(polar.strong_wave_angle(theta)):9.4f} deg")

    # --- All algebraic roots, with the ordered admissibility filter --------
    print("\n  all roots of the wave-angle cubic at this deflection")
    print("      tan(beta)      beta[deg]        sigma  attached  above_CJ  high-comp")
    for root in polar.wave_angle_roots(theta):
        print(f"    {root.t:11.6f} {deg(root.beta):12.4f} {root.sigma:12.5f}"
              f" {str(root.attached):>9} {str(root.above_cj):>9} {str(root.high_compression):>10}")

    # --- Sweeping the wedge angle -----------------------------------------
    print("\n  weak-branch sweep")
    print("    theta[deg]   beta[deg]   downstream")
    for frac in (0.05, 0.25, 0.50, 0.75, 0.95):
        th = cj.theta + frac * (det.theta_max - cj.theta)
        beta = polar.weak_wave_angle(th)
        regime = "supersonic" if th < son.theta else "subsonic"
        print(f"    {deg(th):10.4f} {deg(beta):11.4f}   {regime}")

    # A deflection above theta_max has no attached equilibrium solution.
    try:
        polar.weak_wave_angle(det.theta_max + math.radians(1.0))
    except ValueError as exc:
        print(f"\n  beyond detachment: {exc}")


if __name__ == "__main__":
    main()
