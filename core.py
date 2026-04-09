from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix


MODS = ["pam", "pam", "pam", "psk", "psk", "psk", "qam", "qam", "qam"]
NIVELES = [2, 4, 8, 4, 8, 16, 16, 64, 256]
CODE_TYPES = ["NoCoding", "Hamming", "Conv.", "LDPC", "RS"]
CHANNEL_TYPES = ["AWGN", "Rayleigh", "MIMO", "Vehicular"]
ANT_OPTIONS = [2, 4, 8]
LDPC_MAX_ITERS = int(os.getenv("HESCOD_LDPC_MAX_ITERS", "8"))
LDPC_MAX_BITS_BER = int(os.getenv("HESCOD_LDPC_MAX_BITS_BER", "20000"))
RS_MAX_ML_CODEWORDS = int(os.getenv("HESCOD_RS_MAX_ML_CODEWORDS", "65536"))
CONV_MAX_BITS_BER = int(os.getenv("HESCOD_CONV_MAX_BITS_BER", "300000"))


@dataclass
class TxParams:
    mod: str
    niveles: int
    order: str
    codeType: str
    channel: str
    info: Optional[Any] = None
    legendExtra: str = ""
    legendMIMO: str = ""
    nT: Optional[int] = None
    no: float = 0.0
    snr: float = 0.0


def _mat_candidates() -> List[Path]:
    base = Path(__file__).resolve().parent
    return [base / "ieee802_16e_matrices.mat", base.parent / "ieee802_16e_matrices.mat"]


def _mat_path() -> Path:
    candidates = _mat_candidates()
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def ldpc_matrices_available() -> bool:
    return any(p.exists() for p in _mat_candidates())


def load_ldpc_bases() -> Dict[str, np.ndarray]:
    if not ldpc_matrices_available():
        return {}
    data = loadmat(_mat_path())
    keys = ["ieee802_16e_1_2", "ieee802_16e_2_3A", "ieee802_16e_3_4A", "ieee802_16e_5_6"]
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k in data:
            out[k] = np.array(data[k], dtype=np.int16)
    return out


_LDPC_BASES = load_ldpc_bases()
_LDPC_KEYS = ["ieee802_16e_1_2", "ieee802_16e_2_3A", "ieee802_16e_3_4A", "ieee802_16e_5_6"]
_LDPC_RATE_NAMES = ["1/2", "2/3", "3/4", "5/6"]
_LDPC_BASE_ORDER = [
    _LDPC_BASES.get(_LDPC_KEYS[0]),
    _LDPC_BASES.get(_LDPC_KEYS[1]),
    _LDPC_BASES.get(_LDPC_KEYS[2]),
    _LDPC_BASES.get(_LDPC_KEYS[3]),
]


def system_parameters(
    modulations: Sequence[bool],
    order: str,
    coding: Sequence[bool],
    infoCoding: Sequence[Any],
    channelType: Sequence[bool],
    numAntennas: Sequence[bool],
) -> List[TxParams]:
    ind_mod = [i for i, x in enumerate(modulations) if bool(x)]
    ind_cod = [i for i, x in enumerate(coding) if bool(x)]
    ind_ch = [i for i, x in enumerate(channelType) if bool(x)]
    ind_ant = [i for i, x in enumerate(numAntennas) if bool(x)]

    params: List[TxParams] = []

    for i_mod in ind_mod:
        mod = MODS[i_mod]
        m = NIVELES[i_mod]

        for i_cod in ind_cod:
            code_type = CODE_TYPES[i_cod]

            for i_ch in ind_ch:
                channel = CHANNEL_TYPES[i_ch]
                is_mimo = i_ch == 2
                ant_values = [ANT_OPTIONS[i] for i in ind_ant] if is_mimo else [None]

                for n_ant in ant_values:
                    if code_type == "NoCoding":
                        info = None
                        legend_extra = ""
                    elif code_type == "Conv.":
                        info = list(infoCoding[1])
                        legend_extra = ""
                    elif code_type == "Hamming":
                        l_values = [2, 3, 4]
                        selected = [ix for ix, v in enumerate(infoCoding[0]) if bool(v)]
                        for idx in selected:
                            p = TxParams(
                                mod=mod,
                                niveles=m,
                                order=order,
                                codeType=code_type,
                                channel=channel,
                                info=l_values[idx],
                                legendExtra=f"({l_values[idx]})",
                                legendMIMO=f"({n_ant}x{n_ant})" if n_ant else "",
                                nT=n_ant,
                            )
                            params.append(p)
                        continue
                    elif code_type == "LDPC":
                        if mod == "pam" and m > 2:
                            continue
                        selected = [ix for ix, v in enumerate(infoCoding[2]) if bool(v)]
                        for idx in selected:
                            hb = _LDPC_BASE_ORDER[idx]
                            if hb is None:
                                continue
                            p = TxParams(
                                mod=mod,
                                niveles=m,
                                order=order,
                                codeType=code_type,
                                channel=channel,
                                info=hb,
                                legendExtra=f"({_LDPC_RATE_NAMES[idx]})",
                                legendMIMO=f"({n_ant}x{n_ant})" if n_ant else "",
                                nT=n_ant,
                            )
                            params.append(p)
                        continue
                    else:  # RS
                        l_values = [3, 4, 5]
                        selected = [ix for ix, v in enumerate(infoCoding[3]) if bool(v)]
                        k_rs = int(infoCoding[4])
                        for idx in selected:
                            l = l_values[idx]
                            p = TxParams(
                                mod=mod,
                                niveles=m,
                                order=order,
                                codeType=code_type,
                                channel=channel,
                                info=[l, k_rs],
                                legendExtra=f"({l})",
                                legendMIMO=f"({n_ant}x{n_ant})" if n_ant else "",
                                nT=n_ant,
                            )
                            params.append(p)
                        continue

                    p = TxParams(
                        mod=mod,
                        niveles=m,
                        order=order,
                        codeType=code_type,
                        channel=channel,
                        info=info,
                        legendExtra=legend_extra,
                        legendMIMO=f"({n_ant}x{n_ant})" if n_ant else "",
                        nT=n_ant,
                    )
                    params.append(p)

    return params


def _gray_code(i: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(i, i >> 1)


def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    k = bits.shape[1]
    weights = (1 << np.arange(k - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
    return (bits.astype(np.int64) * weights).sum(axis=1)


def _int_to_bits(vals: np.ndarray, k: int) -> np.ndarray:
    vals = vals.astype(np.int64).reshape(-1, 1)
    shifts = np.arange(k - 1, -1, -1, dtype=np.int64).reshape(1, -1)
    return ((vals >> shifts) & 1).astype(np.uint8)


def _qam_symbols(m: int, order: str) -> np.ndarray:
    k = int(np.log2(m))
    side = int(np.sqrt(m))
    if side * side != m:
        raise ValueError(f"QAM size must be square, got {m}")

    axis_levels = np.arange(-(side - 1), side, 2, dtype=float)
    bits = _int_to_bits(np.arange(m), k)
    half = k // 2
    i_idx = _bits_to_int(bits[:, :half])
    q_idx = _bits_to_int(bits[:, half:])

    if order == "gray":
        i_idx = _gray_code(i_idx)
        q_idx = _gray_code(q_idx)

    i_vals = axis_levels[i_idx]
    q_vals = axis_levels[q_idx]
    return i_vals + 1j * q_vals


def constellation_symbols(mod: str, niveles: int, order: str) -> np.ndarray:
    order = order.lower()
    idx = np.arange(niveles, dtype=np.int64)

    if mod == "pam":
        base = np.arange(-(niveles - 1), niveles, 2, dtype=float)
        map_idx = _gray_code(idx) if order == "gray" else idx
        symbols = base[map_idx]
        return symbols[::-1]

    if mod == "psk":
        map_idx = _gray_code(idx) if order == "gray" else idx
        ang = 2.0 * np.pi * map_idx / niveles
        return np.exp(1j * ang)

    if mod == "qam":
        return _qam_symbols(niveles, order)

    raise ValueError(f"Unsupported modulation: {mod}")


def modulate(encodedBits: np.ndarray, params: TxParams) -> Tuple[np.ndarray, int, np.ndarray, int]:
    symbols = constellation_symbols(params.mod, params.niveles, params.order)
    k = int(np.log2(params.niveles))

    bits = encodedBits.astype(np.uint8).ravel()
    sizePadding = int((-len(bits)) % k)
    if sizePadding:
        bits = np.concatenate([bits, np.zeros(sizePadding, dtype=np.uint8)])

    bit_blocks = bits.reshape(-1, k)
    ind = _bits_to_int(bit_blocks)
    modulated = symbols[ind]
    return modulated, k, symbols, sizePadding


def _hard_demod(receivedSymbols: np.ndarray, symbols: np.ndarray, k: int) -> np.ndarray:
    distances = np.abs(receivedSymbols.reshape(-1, 1) - symbols.reshape(1, -1))
    pos = np.argmin(distances, axis=1)
    return _int_to_bits(pos, k).reshape(-1)


def _soft_llr_demod(receivedSymbols: np.ndarray, symbols: np.ndarray, k: int, no: float) -> np.ndarray:
    bit_labels = _int_to_bits(np.arange(len(symbols)), k)
    sigma2 = max(float(no) / 2.0, 1e-12)
    out = np.zeros(receivedSymbols.size * k, dtype=float)

    for i, r in enumerate(receivedSymbols):
        d2 = np.abs(r - symbols) ** 2
        for b in range(k):
            d0 = d2[bit_labels[:, b] == 0]
            d1 = d2[bit_labels[:, b] == 1]
            llr = (d1.min() - d0.min()) / sigma2
            out[i * k + b] = llr

    return out


def demodulate(receivedSymbols: np.ndarray, params: TxParams, symbols: np.ndarray, sizePadding: int) -> np.ndarray:
    k = int(np.log2(params.niveles))

    if params.codeType == "LDPC":
        demodBits = _soft_llr_demod(receivedSymbols, symbols, k, params.no)
        if sizePadding:
            demodBits = demodBits[: demodBits.size - sizePadding]
        return demodBits

    demodBits = _hard_demod(receivedSymbols, symbols, k)
    if sizePadding:
        demodBits = demodBits[: demodBits.size - sizePadding]
    return demodBits.astype(np.uint8)


# ------------------------
# Coding blocks
# ------------------------


def _hamming_positions(l: int) -> Tuple[int, int, List[int], List[int]]:
    n = 2**l - 1
    parity_pos = [2**i for i in range(l)]
    data_pos = [p for p in range(1, n + 1) if p not in parity_pos]
    k = len(data_pos)
    return n, k, parity_pos, data_pos


def _hamming_encode(bits: np.ndarray, l: int) -> Tuple[np.ndarray, int]:
    n, k, parity_pos, data_pos = _hamming_positions(l)
    sizePadding = int((-len(bits)) % k)
    if sizePadding:
        bits = np.concatenate([bits, np.zeros(sizePadding, dtype=np.uint8)])

    blocks = bits.reshape(-1, k)
    nb = blocks.shape[0]
    cw = np.zeros((nb, n + 1), dtype=np.uint8)  # 1-indexed columns
    cw[:, data_pos] = blocks

    for p in parity_pos:
        idx = [j for j in range(1, n + 1) if (j & p) and (j != p)]
        cw[:, p] = np.mod(cw[:, idx].sum(axis=1), 2).astype(np.uint8)

    return cw[:, 1:].reshape(-1).astype(np.uint8), sizePadding


def _hamming_decode(bits: np.ndarray, l: int) -> np.ndarray:
    n, k, parity_pos, data_pos = _hamming_positions(l)
    if len(bits) % n:
        bits = bits[: len(bits) - (len(bits) % n)]

    if bits.size == 0:
        return np.array([], dtype=np.uint8)

    blocks = bits.reshape(-1, n)
    nb = blocks.shape[0]
    cw = np.zeros((nb, n + 1), dtype=np.uint8)
    cw[:, 1:] = blocks

    syndrome = np.zeros(nb, dtype=np.int32)
    for p in parity_pos:
        idx = [j for j in range(1, n + 1) if j & p]
        check = np.mod(cw[:, idx].sum(axis=1), 2).astype(np.int32)
        syndrome += check * p

    err_rows = np.where((syndrome >= 1) & (syndrome <= n))[0]
    if err_rows.size > 0:
        err_cols = syndrome[err_rows]
        cw[err_rows, err_cols] ^= 1

    return cw[:, data_pos].reshape(-1).astype(np.uint8)


def _octal_to_taps(poly_oct: int, m: int) -> np.ndarray:
    poly_val = int(str(int(poly_oct)), 8)
    bits = [int(b) for b in bin(poly_val)[2:]]
    if len(bits) < m:
        bits = [0] * (m - len(bits)) + bits
    return np.array(bits[-m:], dtype=np.uint8)


def _conv_prepare(gens: Sequence[int]) -> Tuple[int, np.ndarray]:
    gens = [int(x) for x in gens]
    m = max(len(bin(int(str(g), 8))[2:]) for g in gens)
    taps = np.stack([_octal_to_taps(g, m) for g in gens], axis=0)
    return m, taps


def _conv_encode(bits: np.ndarray, gens: Sequence[int]) -> np.ndarray:
    m, taps = _conv_prepare(gens)
    mem = m - 1
    state = np.zeros(mem, dtype=np.uint8)
    outputs: List[int] = []

    for b in bits.astype(np.uint8):
        reg = np.concatenate([[b], state]) if mem else np.array([b], dtype=np.uint8)
        for t in taps:
            outputs.append(int(np.bitwise_and(reg, t).sum() % 2))
        if mem:
            state = reg[:-1]

    return np.array(outputs, dtype=np.uint8)


def _state_to_bits(state: int, mem: int) -> np.ndarray:
    if mem == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(mem - 1, -1, -1)
    return ((state >> shifts) & 1).astype(np.uint8)


def _bits_to_state(bits: np.ndarray) -> int:
    s = 0
    for b in bits:
        s = (s << 1) | int(b)
    return int(s)


def _conv_decode(bits: np.ndarray, gens: Sequence[int]) -> np.ndarray:
    m, taps = _conv_prepare(gens)
    mem = m - 1
    n_out = len(gens)
    n_steps = len(bits) // n_out
    if n_steps == 0:
        return np.array([], dtype=np.uint8)
    rx = bits[: n_steps * n_out].reshape(n_steps, n_out).astype(np.uint8)

    n_states = 2**mem if mem > 0 else 1
    next_state = np.zeros((n_states, 2), dtype=np.int32)
    out_bits = np.zeros((n_states, 2, n_out), dtype=np.uint8)

    for s in range(n_states):
        s_bits = _state_to_bits(s, mem)
        for u in [0, 1]:
            reg = np.concatenate([[u], s_bits]) if mem else np.array([u], dtype=np.uint8)
            out_bits[s, u] = (taps @ reg) % 2
            ns_bits = reg[:-1] if mem else np.zeros(0, dtype=np.uint8)
            next_state[s, u] = _bits_to_state(ns_bits)

    # Precompute compact branch symbols and bit-error lookup.
    out_sym = np.zeros((n_states, 2), dtype=np.uint8)
    for s in range(n_states):
        for u in [0, 1]:
            v = 0
            for b in out_bits[s, u]:
                v = (v << 1) | int(b)
            out_sym[s, u] = v

    max_sym = 1 << n_out
    ham = np.zeros((max_sym, max_sym), dtype=np.uint8)
    for a in range(max_sym):
        for b in range(max_sym):
            ham[a, b] = int((a ^ b).bit_count())

    inf = 1e9
    metrics = np.full(n_states, inf, dtype=float)
    metrics[0] = 0.0
    prev_state = np.zeros((n_steps, n_states), dtype=np.int32)
    prev_u = np.zeros((n_steps, n_states), dtype=np.uint8)

    for t in range(n_steps):
        yv = 0
        for b in rx[t]:
            yv = (yv << 1) | int(b)

        new_metrics = np.full(n_states, inf, dtype=float)
        for s in range(n_states):
            base = metrics[s]
            if base >= inf:
                continue
            for u in [0, 1]:
                ns = int(next_state[s, u])
                cand = base + float(ham[yv, int(out_sym[s, u])])
                if cand < new_metrics[ns]:
                    new_metrics[ns] = cand
                    prev_state[t, ns] = s
                    prev_u[t, ns] = u
        metrics = new_metrics

    st = int(np.argmin(metrics))
    decoded = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = prev_u[t, st]
        st = int(prev_state[t, st])

    return decoded


def _mod2_real(x: np.ndarray) -> np.ndarray:
    return np.mod(np.rint(x), 2).astype(np.uint8)


def genHexp(hb: np.ndarray, z: int) -> csr_matrix:
    nr, nc = hb.shape
    H = np.zeros((nr * z, nc * z), dtype=np.uint8)
    eye = np.eye(z, dtype=np.uint8)

    for i in range(nr):
        for j in range(nc):
            v = int(hb[i, j])
            if v >= 0:
                H[i * z : (i + 1) * z, j * z : (j + 1) * z] = np.roll(eye, shift=v, axis=1)

    return csr_matrix(H)


_LDPC_CACHE: Dict[bytes, Dict[str, Any]] = {}


def _ldpc_prepare(hb: np.ndarray, z: int = 80) -> Dict[str, Any]:
    key = hb.tobytes() + bytes([z])
    if key in _LDPC_CACHE:
        return _LDPC_CACHE[key]

    Hs = genHexp(hb, z)
    H = Hs.toarray().astype(np.uint8)
    m, n = H.shape
    k = n - m

    ir = m - z
    ic = k + z

    A = H[:ir, :k]
    B = H[:ir, k:ic]
    T = H[:ir, ic:]
    C = H[ir:, :k]
    E = H[ir:, ic:]

    T_float = T.astype(float)
    # Mirrors MATLAB right-division + left-division behavior.
    ET1 = _mod2_real(E.astype(float) @ np.linalg.inv(T_float))

    rows, cols = Hs.nonzero()
    n_edges = len(rows)
    row_edges: List[List[int]] = [[] for _ in range(m)]
    col_edges: List[List[int]] = [[] for _ in range(n)]
    for e, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        row_edges[r].append(e)
        col_edges[c].append(e)

    cache = {
        "H_sparse": Hs,
        "H_dense": H,
        "m": m,
        "n": n,
        "k": k,
        "z": z,
        "ir": ir,
        "A": A,
        "B": B,
        "T": T,
        "C": C,
        "E": E,
        "ET1": ET1,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "row_edges": row_edges,
        "col_edges": col_edges,
        "n_edges": n_edges,
    }
    _LDPC_CACHE[key] = cache
    return cache


def _ldpc_encode_block(u: np.ndarray, prep: Dict[str, Any]) -> np.ndarray:
    A = prep["A"]
    B = prep["B"]
    C = prep["C"]
    T = prep["T"]
    ET1 = prep["ET1"]

    au = (A @ u) % 2
    cu = (C @ u) % 2
    p1 = (ET1 @ au + cu) % 2
    Tp2 = (au + (B @ p1) % 2) % 2
    p2 = _mod2_real(np.linalg.solve(T.astype(float), Tp2.astype(float)))

    return np.concatenate([u, p1.astype(np.uint8), p2.astype(np.uint8)]).astype(np.uint8)


def _ldpc_decode_block(llr: np.ndarray, prep: Dict[str, Any], max_iter: int = 25) -> np.ndarray:
    m = prep["m"]
    n = prep["n"]
    k = prep["k"]
    rows = prep["rows"]
    cols = prep["cols"]
    row_edges = prep["row_edges"]
    col_edges = prep["col_edges"]
    n_edges = prep["n_edges"]
    Hs: csr_matrix = prep["H_sparse"]

    llr = llr.astype(float)
    Lq = llr[cols].copy()
    Lr = np.zeros(n_edges, dtype=float)

    post = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        for r in range(m):
            eidx = row_edges[r]
            if not eidx:
                continue
            vals = Lq[eidx]
            signs = np.sign(vals)
            signs[signs == 0] = 1.0
            sign_prod = np.prod(signs)
            av = np.abs(vals)

            if len(av) == 1:
                min1 = 0.0
                min2 = 0.0
                argmin = 0
            else:
                argmin = int(np.argmin(av))
                min1 = float(av[argmin])
                tmp = av.copy()
                tmp[argmin] = np.inf
                min2 = float(np.min(tmp))

            for local_idx, e in enumerate(eidx):
                msg_sign = sign_prod * signs[local_idx]
                mag = min2 if local_idx == argmin else min1
                Lr[e] = msg_sign * mag

        for c in range(n):
            eidx = col_edges[c]
            if not eidx:
                post[c] = llr[c]
                continue
            s = float(np.sum(Lr[eidx]))
            post[c] = llr[c] + s
            for e in eidx:
                Lq[e] = llr[c] + s - Lr[e]

        hard = (post < 0).astype(np.uint8)
        syn = (Hs @ hard) % 2
        if np.count_nonzero(syn) == 0:
            return hard[:k]

    return (post < 0).astype(np.uint8)[:k]


_PRIMITIVE_POLY = {
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
}


class ReedSolomon:
    def __init__(self, m: int, n: int, k: int):
        if m not in _PRIMITIVE_POLY:
            raise ValueError(f"Unsupported RS field size m={m}")
        if not (1 <= k < n):
            raise ValueError(f"Invalid RS(n={n}, k={k})")
        self.m = m
        self.n = n
        self.k = k
        self.nsym = n - k
        self.prim = _PRIMITIVE_POLY[m]
        self.gf_size = 2**m
        self.exp, self.log = self._build_tables()
        self.gen = self._generator_poly(self.nsym)
        self._ml_msgs: Optional[np.ndarray] = None
        self._ml_code: Optional[np.ndarray] = None
        self._syn_table_t2: Optional[Dict[Tuple[int, ...], List[Tuple[int, int]]]] = None
        if (self.gf_size**self.k) <= RS_MAX_ML_CODEWORDS:
            self._build_ml_table()
        elif self.nsym // 2 <= 2:
            self._build_t2_syndrome_table()

    def _build_ml_table(self) -> None:
        # For GF(2^3) the full codebook is tiny and enables robust decoding.
        nmsg = self.gf_size**self.k
        msgs = np.zeros((nmsg, self.k), dtype=np.uint8)
        for i in range(nmsg):
            x = i
            for p in range(self.k - 1, -1, -1):
                msgs[i, p] = x % self.gf_size
                x //= self.gf_size

        code = np.zeros((nmsg, self.n), dtype=np.uint8)
        for i in range(nmsg):
            code[i] = np.array(self.encode(msgs[i].tolist()), dtype=np.uint8)

        self._ml_msgs = msgs
        self._ml_code = code

    def _build_t2_syndrome_table(self) -> None:
        t = self.nsym // 2
        if t > 2:
            return

        table: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}

        # Zero-error syndrome.
        table[tuple([0] * self.nsym)] = []

        # Weight-1 errors.
        for p1 in range(self.n):
            for e1 in range(1, self.gf_size):
                err = [0] * self.n
                err[p1] = e1
                syn = tuple(self.syndromes(err))
                table[syn] = [(p1, e1)]

        if t == 2:
            # Weight-2 errors.
            for p1 in range(self.n - 1):
                for p2 in range(p1 + 1, self.n):
                    for e1 in range(1, self.gf_size):
                        for e2 in range(1, self.gf_size):
                            err = [0] * self.n
                            err[p1] = e1
                            err[p2] = e2
                            syn = tuple(self.syndromes(err))
                            table[syn] = [(p1, e1), (p2, e2)]

        self._syn_table_t2 = table

    def _build_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        exp = np.zeros(2 * (self.gf_size - 1), dtype=np.int32)
        log = np.full(self.gf_size, -1, dtype=np.int32)

        x = 1
        for i in range(self.gf_size - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & self.gf_size:
                x ^= self.prim
        exp[self.gf_size - 1 :] = exp[: self.gf_size - 1]
        return exp, log

    def gf_mul(self, x: int, y: int) -> int:
        if x == 0 or y == 0:
            return 0
        return int(self.exp[self.log[x] + self.log[y]])

    def gf_div(self, x: int, y: int) -> int:
        if y == 0:
            raise ZeroDivisionError("RS GF divide by zero")
        if x == 0:
            return 0
        return int(self.exp[(self.log[x] - self.log[y]) % (self.gf_size - 1)])

    def gf_pow(self, x: int, p: int) -> int:
        if p == 0:
            return 1
        if x == 0:
            return 0
        return int(self.exp[(self.log[x] * p) % (self.gf_size - 1)])

    def poly_scale(self, p: Sequence[int], x: int) -> List[int]:
        return [self.gf_mul(int(c), x) for c in p]

    def poly_add(self, p: Sequence[int], q: Sequence[int]) -> List[int]:
        max_len = max(len(p), len(q))
        out = [0] * max_len
        for i in range(max_len):
            a = int(p[i - len(p)]) if i >= max_len - len(p) else 0
            b = int(q[i - len(q)]) if i >= max_len - len(q) else 0
            out[i] = a ^ b
        while len(out) > 1 and out[0] == 0:
            out.pop(0)
        return out

    def poly_mul(self, p: Sequence[int], q: Sequence[int]) -> List[int]:
        out = [0] * (len(p) + len(q) - 1)
        for j, qj in enumerate(q):
            if qj == 0:
                continue
            for i, pi in enumerate(p):
                out[i + j] ^= self.gf_mul(int(pi), int(qj))
        return out

    def poly_eval(self, p: Sequence[int], x: int) -> int:
        y = int(p[0])
        for i in range(1, len(p)):
            y = self.gf_mul(y, x) ^ int(p[i])
        return y

    def _generator_poly(self, nsym: int) -> List[int]:
        g = [1]
        for i in range(nsym):
            g = self.poly_mul(g, [1, self.gf_pow(2, i + 1)])
        return g

    def encode(self, msg: Sequence[int]) -> List[int]:
        if len(msg) != self.k:
            raise ValueError("RS message block has wrong length")
        out = list(int(x) for x in msg) + [0] * self.nsym
        for i in range(self.k):
            coef = out[i]
            if coef == 0:
                continue
            for j, gj in enumerate(self.gen):
                out[i + j] ^= self.gf_mul(gj, coef)
        parity = out[self.k :]
        return list(int(x) for x in msg) + parity

    def syndromes(self, cw: Sequence[int]) -> List[int]:
        return [self.poly_eval(cw, self.gf_pow(2, i + 1)) for i in range(self.nsym)]

    def _find_error_locator(self, synd: Sequence[int]) -> List[int]:
        C = [1] + [0] * self.nsym
        B = [1] + [0] * self.nsym
        L = 0
        m = 1
        b = 1

        for n in range(self.nsym):
            d = int(synd[n])
            for i in range(1, L + 1):
                d ^= self.gf_mul(C[i], int(synd[n - i]))

            if d == 0:
                m += 1
                continue

            T = C.copy()
            coef = self.gf_div(d, b)
            for i in range(m, self.nsym + 1):
                C[i] ^= self.gf_mul(coef, B[i - m])

            if 2 * L <= n:
                L = n + 1 - L
                B = T
                b = d
                m = 1
            else:
                m += 1

        while len(C) > 1 and C[-1] == 0:
            C.pop()
        return C

    def _find_error_evaluator(self, synd: Sequence[int], err_loc: Sequence[int]) -> List[int]:
        # (S(x) * Lambda(x)) mod x^(nsym)
        synd_poly = list(synd)
        prod = self.poly_mul(synd_poly, err_loc)
        return prod[: self.nsym]

    def _find_errors(self, err_loc: Sequence[int], nmess: int) -> Optional[List[int]]:
        errs = len(err_loc) - 1
        err_pos: List[int] = []
        for i in range(nmess):
            if self.poly_eval(err_loc, self.gf_pow(2, i)) == 0:
                err_pos.append(nmess - 1 - i)
        if len(err_pos) != errs:
            return None
        return err_pos

    def _correct_errata(self, cw: List[int], synd: Sequence[int], err_pos: Sequence[int]) -> List[int]:
        coef_pos = [len(cw) - 1 - p for p in err_pos]
        err_loc = [1]
        for cp in coef_pos:
            err_loc = self.poly_mul(err_loc, [1, self.gf_pow(2, cp)])

        err_eval = self._find_error_evaluator(synd[::-1], err_loc[::-1])[::-1]

        X = [self.gf_pow(2, -cp % (self.gf_size - 1)) for cp in coef_pos]

        E = [0] * len(cw)
        for i, Xi in enumerate(X):
            Xi_inv = self.gf_div(1, Xi)
            err_loc_prime = 1
            for j, Xj in enumerate(X):
                if i != j:
                    err_loc_prime = self.gf_mul(err_loc_prime, 1 ^ self.gf_mul(Xi_inv, Xj))
            y = self.poly_eval(err_eval[::-1], Xi_inv)
            magnitude = self.gf_div(self.gf_mul(self.gf_pow(Xi, 1), y), err_loc_prime)
            E[err_pos[i]] = magnitude

        return [c ^ e for c, e in zip(cw, E)]

    def decode(self, cw: Sequence[int]) -> List[int]:
        if len(cw) != self.n:
            raise ValueError("RS codeword has wrong length")
        cw_list = [int(x) for x in cw]

        if self._ml_code is not None and self._ml_msgs is not None:
            cw_arr = np.array(cw_list, dtype=np.uint8).reshape(1, -1)
            d = np.count_nonzero(self._ml_code != cw_arr, axis=1)
            best = int(np.argmin(d))
            return self._ml_msgs[best].astype(np.int64).tolist()

        if self._syn_table_t2 is not None:
            syn = tuple(self.syndromes(cw_list))
            corr = self._syn_table_t2.get(syn)
            if corr is not None:
                out = cw_list.copy()
                for p, e in corr:
                    out[p] ^= int(e)
                if max(self.syndromes(out), default=0) == 0:
                    return out[: self.k]

        synd = self.syndromes(cw_list)
        if max(synd, default=0) == 0:
            return cw_list[: self.k]

        err_loc = self._find_error_locator(synd)
        err_pos = self._find_errors(err_loc[::-1], len(cw_list))
        if err_pos is None:
            return cw_list[: self.k]

        corrected = self._correct_errata(cw_list, synd, err_pos)
        synd2 = self.syndromes(corrected)
        if max(synd2, default=0) != 0:
            return cw_list[: self.k]
        return corrected[: self.k]


_RS_CACHE: Dict[Tuple[int, int], ReedSolomon] = {}


def _rs_codec(m: int, k: int) -> ReedSolomon:
    key = (m, k)
    if key not in _RS_CACHE:
        n = 2**m - 1
        _RS_CACHE[key] = ReedSolomon(m=m, n=n, k=k)
    return _RS_CACHE[key]


def encodingOp(sourceBits: np.ndarray, params: TxParams) -> Tuple[np.ndarray, float, int]:
    bits = sourceBits.astype(np.uint8).ravel()
    t = params.codeType
    sizePadding = 0

    if t == "NoCoding":
        return bits.copy(), 1.0, 0

    if t == "Hamming":
        l = int(params.info)
        enc, sizePadding = _hamming_encode(bits, l)
        n, k, _, _ = _hamming_positions(l)
        return enc, float(k / n), sizePadding

    if t == "Conv.":
        info = [int(x) for x in params.info]
        enc = _conv_encode(bits, info)
        return enc, 1.0 / float(len(info)), 0

    if t == "LDPC":
        prep = _ldpc_prepare(np.array(params.info, dtype=np.int16), z=80)
        k = prep["k"]
        n = prep["n"]
        sizePadding = int((-len(bits)) % k)
        if sizePadding:
            bits = np.concatenate([bits, np.zeros(sizePadding, dtype=np.uint8)])

        nb = len(bits) // k
        out = np.zeros(nb * n, dtype=np.uint8)
        for i in range(nb):
            u = bits[i * k : (i + 1) * k]
            out[i * n : (i + 1) * n] = _ldpc_encode_block(u, prep)
        return out, float(k / n), sizePadding

    # RS
    l = int(params.info[0])
    k_rs = int(params.info[1])
    n_rs = 2**l - 1
    block_bits = l * k_rs
    sizePadding = int((-len(bits)) % block_bits)
    if sizePadding:
        bits = np.concatenate([bits, np.zeros(sizePadding, dtype=np.uint8)])

    codec = _rs_codec(l, k_rs)
    out_blocks: List[np.ndarray] = []
    for b in bits.reshape(-1, block_bits):
        msg_symbols = _bits_to_int(b.reshape(-1, l))
        cw_symbols = np.array(codec.encode(msg_symbols.tolist()), dtype=np.int64)
        cw_bits = _int_to_bits(cw_symbols, l).reshape(-1)
        out_blocks.append(cw_bits)

    out = np.concatenate(out_blocks).astype(np.uint8)
    return out, float(k_rs / n_rs), sizePadding


def decodingOp(demodulatedBits: np.ndarray, params: TxParams, sizePadding: int) -> np.ndarray:
    t = params.codeType

    if t == "NoCoding":
        est = demodulatedBits.astype(np.uint8).ravel()

    elif t == "Hamming":
        l = int(params.info)
        est = _hamming_decode(demodulatedBits.astype(np.uint8).ravel(), l)

    elif t == "Conv.":
        info = [int(x) for x in params.info]
        est = _conv_decode(demodulatedBits.astype(np.uint8).ravel(), info)

    elif t == "LDPC":
        prep = _ldpc_prepare(np.array(params.info, dtype=np.int16), z=80)
        n = prep["n"]
        k = prep["k"]
        llr = demodulatedBits.astype(float).ravel()
        nb = len(llr) // n
        out = np.zeros(nb * k, dtype=np.uint8)
        for i in range(nb):
            block = llr[i * n : (i + 1) * n]
            out[i * k : (i + 1) * k] = _ldpc_decode_block(block, prep, max_iter=LDPC_MAX_ITERS)
        est = out

    else:  # RS
        l = int(params.info[0])
        k_rs = int(params.info[1])
        n_rs = 2**l - 1
        bits = demodulatedBits.astype(np.uint8).ravel()
        block_bits = l * n_rs
        nb = len(bits) // block_bits
        codec = _rs_codec(l, k_rs)

        out_blocks: List[np.ndarray] = []
        for i in range(nb):
            b = bits[i * block_bits : (i + 1) * block_bits]
            cw_symbols = _bits_to_int(b.reshape(-1, l))
            msg_symbols = np.array(codec.decode(cw_symbols.tolist()), dtype=np.int64)
            msg_bits = _int_to_bits(msg_symbols, l).reshape(-1)
            out_blocks.append(msg_bits)
        est = np.concatenate(out_blocks).astype(np.uint8) if out_blocks else np.array([], dtype=np.uint8)

    if sizePadding:
        est = est[: est.size - sizePadding]
    return est.astype(np.uint8)


# ------------------------
# Channel and system functions
# ------------------------


def channelTx(modulatedSymbols: np.ndarray, no: float, params: TxParams) -> np.ndarray:
    t = params.channel
    snr = float(params.snr)
    m = float(params.niveles)
    s = modulatedSymbols

    if t == "AWGN":
        if np.isrealobj(s):
            noise = np.sqrt(no / 2.0) * np.random.randn(*s.shape)
        else:
            noise = np.sqrt(no / 2.0) * (
                np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape)
            )
        return s + noise

    if t == "Rayleigh":
        dopplerS = np.sqrt((snr + 1.0) / np.log2(m))
        ch = dopplerS * np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape)
        channelSymbols = ch * s
        noise = np.sqrt(no / 2.0) * (
            np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape)
        )
        noiseSymbols = channelSymbols + noise
        zf = np.conj(ch) / np.maximum(np.abs(ch) ** 2, 1e-12)
        return zf * noiseSymbols

    if t == "MIMO":
        nT = int(params.nT or 2)
        nR = nT
        nSymbols = s.size
        block_size = 1000
        out = np.zeros(nSymbols, dtype=complex)

        for start in range(0, nSymbols, block_size):
            end = min(start + block_size, nSymbols)
            block = s[start:end]
            H = (np.random.randn(nR, nT) + 1j * np.random.randn(nR, nT)) / np.sqrt(2.0)
            U, _, _ = np.linalg.svd(H)
            p = U[0, :].reshape(-1, 1)
            hp = H @ p
            tx = hp @ block.reshape(1, -1)
            noise = np.sqrt(no / 2.0) * (
                np.random.randn(nR, block.size) + 1j * np.random.randn(nR, block.size)
            )
            rx = tx + noise

            a = (hp.conj().T @ hp + no / 2.0)
            filt = (1.0 / a) * hp.conj().T
            out[start:end] = (filt @ rx).reshape(-1)

        return out

    # Vehicular (time-varying Rayleigh approximation)
    fD = 120.0
    phase_inc = np.sqrt(2 * np.pi * fD / 1e4) * np.random.randn(s.size)
    phase = np.cumsum(phase_inc)
    ray = (np.random.randn(s.size) + 1j * np.random.randn(s.size)) / np.sqrt(2.0)
    ch = ray * np.exp(1j * phase)
    ch_symbols = ch * s
    noise = np.sqrt(no / 2.0) * (np.random.randn(s.size) + 1j * np.random.randn(s.size))
    noisy = ch_symbols + noise
    zf = np.conj(ch) / np.maximum(np.abs(ch) ** 2, 1e-12)
    return zf * noisy


def calcular_ber(
    sourceBits: np.ndarray,
    SNRdB: Sequence[float],
    params: TxParams,
    showConst: bool,
    progress_step: Optional[Callable[[int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
    snrs = 10.0 ** (np.array(SNRdB, dtype=float) / 10.0)
    ber = np.zeros(len(snrs), dtype=float)
    m_hist: List[np.ndarray] = []
    r_hist: List[np.ndarray] = []
    symbols_out = np.array([], dtype=complex)

    source = sourceBits.astype(np.uint8).ravel()
    if params.codeType == "LDPC" and source.size > LDPC_MAX_BITS_BER:
        source = source[:LDPC_MAX_BITS_BER]
    if params.codeType == "Conv." and source.size > CONV_MAX_BITS_BER:
        source = source[:CONV_MAX_BITS_BER]
    enc, rate, size1 = encodingOp(source, params)
    modsym, k, symbols, size2 = modulate(enc, params)
    symbols_out = symbols

    Es = float(np.mean(np.abs(modsym) ** 2))
    Eb = Es / k

    for j, snr_lin in enumerate(snrs):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Simulation cancelled")

        no = (Eb / snr_lin) / rate

        params.no = no
        params.snr = float(SNRdB[j])

        recv = channelTx(modsym, no, params)
        dem = demodulate(recv, params, symbols, size2)
        est = decodingOp(dem, params, size1)

        ncomp = min(source.size, est.size)
        if ncomp == 0:
            ber[j] = 1.0
            err_count = ncomp
        else:
            err_count = int(np.count_nonzero(source[:ncomp] != est[:ncomp]))
            if err_count == 0:
                # On log-scale BER plots, 0 produces a vertical drop to -inf.
                # Use a finite plotting lower bound when no errors are observed.
                ber[j] = 0.5 / float(ncomp)
            else:
                ber[j] = float(err_count / ncomp)

        if showConst:
            m_hist.append(modsym)
            r_hist.append(recv)

        if (err_count == 0 or ber[j] < 1e-7) and not showConst:
            ber = ber[: j + 1]
            if progress_step is not None:
                progress_step(1)
            break

        if progress_step is not None:
            progress_step(1)

    return ber, m_hist, r_hist, symbols_out


def getCapacity(
    channelType: Sequence[bool],
    minSNR: float,
    maxSNR: float,
    numAntennas: Sequence[bool],
) -> Tuple[float, float]:
    ind = [i for i, x in enumerate(channelType) if bool(x)]
    if not ind:
        return 0.0, 0.0

    t = CHANNEL_TYPES[ind[0]]
    ants = [ANT_OPTIONS[i] for i, x in enumerate(numAntennas) if bool(x)]
    n_ant = ants[0] if ants else 2

    snr1 = 10.0 ** (minSNR / 10.0)
    snr2 = 10.0 ** (maxSNR / 10.0)
    ch_size = 100_000

    if t == "AWGN":
        return float(np.log2(1 + snr1)), float(np.log2(1 + snr2))

    if t in ("Rayleigh", "Vehicular"):
        h = (np.random.randn(ch_size) + 1j * np.random.randn(ch_size)) / np.sqrt(2.0)
        cap1 = np.log2(np.mean(1 + snr1 * np.abs(h) ** 2))
        cap2 = np.log2(np.mean(1 + snr2 * np.abs(h) ** 2))
        return float(cap1), float(cap2)

    # MIMO
    nT = n_ant
    nR = n_ant
    cap1 = np.zeros(ch_size, dtype=float)
    cap2 = np.zeros(ch_size, dtype=float)
    for i in range(ch_size):
        H = (np.random.randn(nR, nT) + 1j * np.random.randn(nR, nT)) / np.sqrt(2.0)
        sd = np.linalg.svd(H, compute_uv=False)
        p = np.sum(sd**2)
        cap1[i] = np.log2(1 + snr1 / nT * p)
        cap2[i] = np.log2(1 + snr2 / nT * p)
    return float(cap1.mean()), float(cap2.mean())


def getParameters(
    modulation: Sequence[bool],
    codingMethod: Sequence[bool],
    infoCoding: Sequence[Any],
    numAntennas: Sequence[bool],
) -> Tuple[float, int, float, float, int]:
    dist_ldpc = [12, 9, 6, 4]
    warning = 0

    ind1 = [i for i, x in enumerate(modulation) if bool(x)]
    if not ind1:
        return 0.0, 0, 0.0, 0.0, 0

    m = NIVELES[ind1[0]]
    mod = MODS[ind1[0]]

    ants = [ANT_OPTIONS[i] for i, x in enumerate(numAntennas) if bool(x)]
    n_ant = ants[0] if ants else None

    k_mod = float(np.log2(m))
    if mod == "pam":
        Es = (m * m - 1) / 3.0
    elif mod == "psk":
        Es = 1.0
    else:
        symbols = constellation_symbols("qam", m, "gray")
        Es = float(np.mean(np.abs(symbols) ** 2))

    Eb = Es / k_mod

    ind2 = [i for i, x in enumerate(codingMethod) if bool(x)]
    if not ind2:
        return Eb, 0, 0.0, 0.0, warning

    c = ind2[0]
    minDist = 0
    rate = 1.0

    if c == 0:
        minDist = 0
        rate = 1.0

    elif c == 1:
        minDist = 3
        lvals = [2, 3, 4]
        selected = [i for i, v in enumerate(infoCoding[0]) if bool(v)]
        l = lvals[selected[0]] if selected else 2
        n = 2**l - 1
        k = n - l
        rate = k / n

    elif c == 2:
        info = [int(x) for x in infoCoding[1]]
        _, taps = _conv_prepare(info)
        m_conv = taps.shape[1]
        # Approximation to MATLAB distspec result.
        minDist = max(2, m_conv)
        rate = 1.0 / len(info)

    elif c == 3:
        selected = [i for i, v in enumerate(infoCoding[2]) if bool(v)]
        idx = selected[0] if selected else 0
        rv = [1 / 2, 2 / 3, 3 / 4, 5 / 6]
        rate = rv[idx]
        minDist = dist_ldpc[idx]

    else:  # RS
        mvals = [3, 4, 5]
        selected = [i for i, v in enumerate(infoCoding[3]) if bool(v)]
        m_rs = mvals[selected[0]] if selected else 3
        k_rs = int(infoCoding[4])
        n_rs = 2**m_rs - 1
        minDist = n_rs - k_rs + 1
        rate = k_rs / n_rs

    tasa = (np.log2(m) * rate) * (n_ant if n_ant is not None else 1)
    return float(Eb), int(minDist), float(tasa), float(rate), warning


def build_legend(params: TxParams) -> str:
    ord_name = params.order
    ord_title = ord_name[0].upper() + ord_name[1:]
    return (
        f"{params.codeType} {params.legendExtra} + {params.niveles}-{params.mod.upper()} "
        f"({ord_title}) -- [{params.channel}]{params.legendMIMO}"
    )


def simulate_system(
    sourceBits: np.ndarray,
    modulations: Sequence[bool],
    order: str,
    coding: Sequence[bool],
    infoCoding: Sequence[Any],
    channelType: Sequence[bool],
    numAntennas: Sequence[bool],
    SNRdB: Sequence[float],
    showConst: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    params_all = system_parameters(modulations, order, coding, infoCoding, channelType, numAntennas)
    out: List[Dict[str, Any]] = []
    thr = 1e-4
    total_steps = max(1, len(params_all) * max(1, len(SNRdB)))
    done_steps = 0

    if progress_callback is not None:
        progress_callback(0, total_steps, "Preparando simulacion...")

    for idx, p in enumerate(params_all, start=1):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Simulation cancelled")

        status = f"Simulando {idx}/{len(params_all)}: {build_legend(p)}"
        if p.codeType == "LDPC" and sourceBits.size > LDPC_MAX_BITS_BER:
            status += f" (muestreo {LDPC_MAX_BITS_BER} bits)"
        if p.codeType == "Conv." and sourceBits.size > CONV_MAX_BITS_BER:
            status += f" (muestreo {CONV_MAX_BITS_BER} bits)"
        if progress_callback is not None:
            progress_callback(done_steps, total_steps, status)

        def _step(inc: int = 1) -> None:
            nonlocal done_steps
            done_steps = min(total_steps, done_steps + inc)
            if progress_callback is not None:
                progress_callback(done_steps, total_steps, status)

        base_snr = np.array(SNRdB, dtype=float)
        ber, m_hist, r_hist, symbols = calcular_ber(
            sourceBits,
            base_snr,
            p,
            showConst,
            progress_step=_step,
            cancel_check=cancel_check,
        )
        snr_plot = base_snr[: len(ber)]

        if p.codeType == "LDPC":
            pos = np.where(ber < thr)[0]
            if pos.size > 0:
                pivot = snr_plot[pos[0]]
                fine = np.arange(pivot - 0.8, pivot + 0.4 + 1e-9, 0.1)
                fine_ber, _, _, _ = calcular_ber(
                    sourceBits,
                    fine,
                    p,
                    showConst,
                    cancel_check=cancel_check,
                )
                ber = np.concatenate([ber[: pos[0]], fine_ber])
                snr_plot = np.concatenate([snr_plot[: pos[0]], fine[: len(fine_ber)]])

        out.append(
            {
                "params": p,
                "ber": ber,
                "snr": snr_plot,
                "legend": build_legend(p),
                "modulated_hist": m_hist,
                "received_hist": r_hist,
                "symbols": symbols,
            }
        )

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, "Simulacion completada")

    return out


def _bits_from_image_y(path: Path) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    from PIL import Image

    img = Image.open(path)
    ycbcr = img.convert("YCbCr")
    y = np.array(ycbcr)[:, :, 0].astype(np.uint8)
    bits = np.unpackbits(y.reshape(-1), bitorder="big").astype(np.uint8)
    return bits, y.shape, y


def load_image_bits(path: Path) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    return _bits_from_image_y(path)


def bits_to_image(bits: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    needed = n * 8
    if bits.size < needed:
        bits = np.concatenate([bits, np.zeros(needed - bits.size, dtype=np.uint8)])
    else:
        bits = bits[:needed]
    packed = np.packbits(bits.reshape(-1, 8), axis=1, bitorder="big").reshape(-1)
    return packed.reshape(shape).astype(np.uint8)


def simulate_image(
    sourceBits: np.ndarray,
    image_shape: Tuple[int, int],
    modulations: Sequence[bool],
    order: str,
    coding: Sequence[bool],
    infoCoding: Sequence[Any],
    channelType: Sequence[bool],
    numAntennas: Sequence[bool],
    SNRdB: Sequence[float],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    params_all = system_parameters(modulations, order, coding, infoCoding, channelType, numAntennas)
    snr_vec = np.array(SNRdB, dtype=float)
    out: List[Dict[str, Any]] = []
    total_steps = max(1, len(params_all) * max(1, len(snr_vec)))
    done_steps = 0

    if progress_callback is not None:
        progress_callback(0, total_steps, "Preparando simulacion de imagen...")

    for idx, p in enumerate(params_all, start=1):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Simulation cancelled")
        status = f"Imagen {idx}/{len(params_all)}: {build_legend(p)}"
        for snr_db in snr_vec:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Simulation cancelled")
            enc, rate, size1 = encodingOp(sourceBits, p)
            modsym, k, symbols, size2 = modulate(enc, p)
            Es = float(np.mean(np.abs(modsym) ** 2))
            Eb = Es / k
            no = (Eb / (10.0 ** (snr_db / 10.0))) / rate
            p.no = no
            p.snr = float(snr_db)
            recv = channelTx(modsym, no, p)
            dem = demodulate(recv, p, symbols, size2)
            est = decodingOp(dem, p, size1)
            out.append(
                {
                    "params": p,
                    "snr": snr_db,
                    "rx_image": bits_to_image(est.astype(np.uint8), image_shape),
                    "tx_image": bits_to_image(sourceBits.astype(np.uint8), image_shape),
                }
            )

            done_steps = min(total_steps, done_steps + 1)
            if progress_callback is not None:
                progress_callback(done_steps, total_steps, status)

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, "Simulacion completada")

    return out
