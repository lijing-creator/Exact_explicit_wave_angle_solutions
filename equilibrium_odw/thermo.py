"""NASA Glenn nine-coefficient thermodynamics for ideal-gas mixtures.

The species data are read from a user-supplied file in the NASA-9 format of
McBride, Zehe & Gordon (NASA TP-2002-211556), for example the ``thermo.inp``
library of NASA CEA or ``thermo_CT.inp`` of Combustion Toolbox.  No data file
is distributed with this package.

    Cp0/Ru     = a1/T^2 + a2/T + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
    H0/(Ru T)  = -a1/T^2 + a2 lnT/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4
                 + a7 T^4/5 + b1/T
    S0/Ru      = -a1/(2T^2) - a2/T + a3 lnT + a4 T + a5 T^2/2 + a6 T^3/3
                 + a7 T^4/4 + b2
    G0         = H0 - T S0

Mixture properties are mole-fraction weighted; specific quantities use the
mixture molar mass.  Enthalpies include the enthalpy of formation, so the
chemical energy release enters the energy balance automatically.
"""

import numpy as np

RU = 8.31446261815324          # J/(mol K), CODATA 2018
P_REF = 1.0e5                  # Pa; standard-state pressure of the NASA-9 fits (1 bar)


def _f(s):
    """Parse a Fortran real, which may use the D exponent."""
    s = s.strip()
    if not s:
        return 0.0
    return float(s.replace("D", "E").replace("d", "e"))


class Species(object):
    """Piecewise nine-coefficient fit for one species."""

    def __init__(self, name, M_kg_per_mol, hf298, intervals, composition=None):
        self.name = name
        self.M = M_kg_per_mol                   # kg/mol
        self.hf298 = hf298                      # J/mol
        self.composition = composition or {}    # {element: atoms per molecule}
        self.intervals = intervals              # [(Tlo, Thi, a[7], b1, b2), ...]
        self.Tmin = min(iv[0] for iv in intervals)
        self.Tmax = max(iv[1] for iv in intervals)

    def _pick(self, T):
        """Coefficients of the interval containing T; outside the fit, the nearest one."""
        for lo, hi, a, b1, b2 in self.intervals:
            if lo - 1e-9 <= T <= hi + 1e-9:
                return a, b1, b2
        if T < self.Tmin:
            lo, hi, a, b1, b2 = self.intervals[0]
        else:
            lo, hi, a, b1, b2 = self.intervals[-1]
        return a, b1, b2

    def cp0(self, T):
        """Molar heat capacity, J/(mol K)."""
        a, _, _ = self._pick(T)
        return RU * (a[0] / T**2 + a[1] / T + a[2] + a[3] * T
                     + a[4] * T**2 + a[5] * T**3 + a[6] * T**4)

    def h0(self, T):
        """Molar enthalpy including the enthalpy of formation, J/mol."""
        a, b1, _ = self._pick(T)
        return RU * (-a[0] / T + a[1] * np.log(T) + a[2] * T + a[3] * T**2 / 2.0
                     + a[4] * T**3 / 3.0 + a[5] * T**4 / 4.0 + a[6] * T**5 / 5.0
                     + b1)

    def s0(self, T):
        """Molar entropy at the standard-state pressure, J/(mol K)."""
        a, _, b2 = self._pick(T)
        return RU * (-a[0] / (2.0 * T**2) - a[1] / T + a[2] * np.log(T)
                     + a[3] * T + a[4] * T**2 / 2.0 + a[5] * T**3 / 3.0
                     + a[6] * T**4 / 4.0 + b2)

    def g0(self, T):
        """Molar Gibbs energy at the standard-state pressure, J/mol."""
        return self.h0(T) - T * self.s0(T)


def load_nasa9(path):
    """Read a NASA-9 thermodynamic library and return ``{name: Species}``."""
    with open(path, "r", errors="replace") as fh:
        raw = fh.readlines()
    lines = [ln.rstrip("\n") for ln in raw
             if not ln.startswith("!") and ln.strip() and not ln.startswith("#")]

    out, i = {}, 0
    while i < len(lines):
        head = lines[i]
        # a record starts with the species name in columns 1-18
        if head.startswith(" ") or len(head) < 2:
            i += 1
            continue
        name = head[:18].strip()
        if i + 1 >= len(lines):
            break
        info = lines[i + 1]
        try:
            n_int = int(info[:2])
        except ValueError:
            i += 1
            continue
        try:
            M = _f(info[52:65]) * 1e-3          # g/mol -> kg/mol
            hf = _f(info[65:80])
        except ValueError:
            i += 1
            continue
        # formula: columns 11-50, five (A2 symbol, F6.2 count) pairs
        comp = {}
        for k in range(5):
            fld = info[10 + 8 * k:18 + 8 * k]
            if len(fld) < 8:
                break
            sym = fld[:2].strip()
            try:
                cnt = _f(fld[2:])
            except ValueError:
                continue
            if sym and cnt != 0.0:
                comp[sym] = comp.get(sym, 0.0) + cnt

        intervals, j, ok = [], i + 2, True
        for _ in range(max(n_int, 0)):
            if j + 2 >= len(lines) + 1 or j + 2 > len(lines):
                ok = False
                break
            rng, c1, c2 = lines[j], lines[j + 1], lines[j + 2]
            try:
                Tlo, Thi = _f(rng[:11]), _f(rng[11:22])
                a = [_f(c1[k * 16:(k + 1) * 16]) for k in range(5)]
                a += [_f(c2[0:16]), _f(c2[16:32])]
                b1, b2 = _f(c2[48:64]), _f(c2[64:80])
            except (ValueError, IndexError):
                ok = False
                break
            intervals.append((Tlo, Thi, a, b1, b2))
            j += 3
        if ok and intervals:
            out[name] = Species(name, M, hf, intervals, comp)
            i = j
        else:
            i += 1
    return out


class Mixture(object):
    """Ideal-gas mixture of a fixed list of species."""

    def __init__(self, species_list):
        self.sp = list(species_list)
        self.names = [s.name for s in self.sp]
        self.Mi = np.array([s.M for s in self.sp])      # kg/mol
        self.n = len(self.sp)
        # element matrix a[i, j]: atoms of element j in species i
        els = []
        for s in self.sp:
            for e in s.composition:
                if e not in els:
                    els.append(e)
        self.elements = sorted(els)
        self.a = np.zeros((self.n, len(self.elements)))
        for i, s in enumerate(self.sp):
            for e, c in s.composition.items():
                self.a[i, self.elements.index(e)] = c

    # -- molar quantities ---------------------------------------------------
    def Cp0_mole(self, T, X):
        return float(np.dot(X, [s.cp0(T) for s in self.sp]))

    def H0_mole(self, T, X):
        return float(np.dot(X, [s.h0(T) for s in self.sp]))

    def S0_mole(self, T, X):
        return float(np.dot(X, [s.s0(T) for s in self.sp]))

    def M_mix(self, X):
        return float(np.dot(X, self.Mi))                # kg/mol

    # -- specific quantities --------------------------------------------------
    def R(self, X):
        """Specific gas constant, J/(kg K)."""
        return RU / self.M_mix(X)

    def cp(self, T, X):
        """Frozen specific heat at constant pressure, J/(kg K)."""
        return self.Cp0_mole(T, X) / self.M_mix(X)

    def h(self, T, X):
        """Specific enthalpy including enthalpies of formation, J/kg."""
        return self.H0_mole(T, X) / self.M_mix(X)

    def g0_over_RuT(self, T):
        """Species G0/(Ru T), used by the Gibbs minimisation."""
        return np.array([s.g0(T) for s in self.sp]) / (RU * T)
