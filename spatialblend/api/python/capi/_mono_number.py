"""Unity/mono double.ToString("R", CultureInfo.InvariantCulture).

Unity 6 runs MonoBleedingEdge, whose "R" is the .NET-Framework algorithm:
G15 when mono's own (lossy) Number.NumberToDouble parses the 15-digit
buffer back to the same double, else G17; digits are the EXACT binary value
rounded half AWAY from zero; scientific notation as .NET "G" (exponent < -5
or >= precision, sign, at least two digits); -0.0 prints as "0".

Ported from the K1 writer of record in the evaluation harness (not
public), which is byte-identical to
Unity's writer on 64 Unity-written goldens and 300k random doubles. The mono
NumberToDouble tables were read from Unity's shipped mscorlib by that work.
Pinned here by tests/data/mono_double_r_sample.tsv, produced by Unity
6000.0.56f1's own mono (tests/data/MonoDoubleR.cs).

Known consequence: Unity's writer is not lossless for every double
(5.0867259815001304 is written as 5.08672598150013, 1 ulp off); this module
reproduces the same bytes.
"""
from decimal import ROUND_HALF_UP, Decimal
import math
import struct
from typing import Tuple


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _fail(message: str) -> None:
    raise ValueError(message)


def format_double(value: float) -> str:
    """``double.ToString("R", CultureInfo.InvariantCulture)`` on Unity's mono.

    G15 when it round-trips, else G17; scientific notation as .NET "G"
    (exponent < -5 or >= precision, sign and at least two digits), which is
    exactly Python's ``%G``. Negative zero prints as ``0``.
    """
    if not _finite(value):
        _fail("non-finite number")
    value = float(value)
    if value == 0.0:
        return "0"
    # The round-trip test is mono's own lossy NumberToDouble on the 15-digit
    # buffer, not an exact parse: 3.6129 prints as 3.6128999999999998 and
    # 5.0867259815001304 as 5.08672598150013 on Unity, and so here.
    sign, digits, adjusted = _rounded_digits(value, 15)
    if _mono_number_to_double(digits, adjusted + 1, sign) == value:
        return _mono_g(value, 15)
    return _mono_g(value, 17)


def _rounded_digits(value: float, precision: int) -> Tuple[int, str, int]:
    """(sign, digits, decimal exponent of the leading digit) of the EXACT
    binary value rounded half away from zero to ``precision`` digits."""
    exact = Decimal(value)
    _, digits, exponent = exact.as_tuple()
    adjusted = len(digits) + exponent - 1
    rounded = exact.quantize(Decimal(1).scaleb(adjusted - precision + 1), rounding=ROUND_HALF_UP)
    sign, digits, exponent = rounded.as_tuple()
    return sign, "".join(map(str, digits)), len(digits) + exponent - 1


# Mono corlib (Unity MonoBleedingEdge unityjit mscorlib.dll) Number.NumberToDouble
# tables, read from the shipped assembly: 10^1..10^15 and 10^-1..10^-15, then
# 10^16k and 10^-16k, as normalized 64-bit mantissas, with their exponents.
_P10 = (0xa000000000000000, 0xc800000000000000, 0xfa00000000000000, 0x9c40000000000000, 0xc350000000000000,
        0xf424000000000000, 0x9896800000000000, 0xbebc200000000000, 0xee6b280000000000, 0x9502f90000000000,
        0xba43b74000000000, 0xe8d4a51000000000, 0x9184e72a00000000, 0xb5e620f480000000, 0xe35fa931a0000000,
        0xcccccccccccccccd, 0xa3d70a3d70a3d70b, 0x83126e978d4fdf3c, 0xd1b71758e219652e, 0xa7c5ac471b478425,
        0x8637bd05af6c69b7, 0xd6bf94d5e57a42be, 0xabcc77118461ceff, 0x89705f4136b4a599, 0xdbe6fecebdedd5c2,
        0xafebff0bcb24ab02, 0x8cbccc096f5088cf, 0xe12e13424bb40e18, 0xb424dc35095cd813, 0x901d7cf73ab0acdc)
_E10 = (4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50)
_P16 = (0x8e1bc9bf04000000, 0x9dc5ada82b70b59e, 0xaf298d050e4395d6, 0xc2781f49ffcfa6d4, 0xd7e77a8f87daf7fa,
        0xefb3ab16c59b14a0, 0x850fadc09923329c, 0x93ba47c980e98cde, 0xa402b9c5a8d3a6e6, 0xb616a12b7fe617a8,
        0xca28a291859bbf90, 0xe070f78d39275566, 0xf92e0c3537826140, 0x8a5296ffe33cc92c, 0x9991a6f3d6bf1762,
        0xaa7eebfb9df9de8a, 0xbd49d14aa79dbc7e, 0xd226fc195c6a2f88, 0xe950df20247c83f8, 0x81842f29f2cce373,
        0x8fcac257558ee4e2, 0xe69594bec44de160, 0xcfb11ead453994c3, 0xbb127c53b17ec165, 0xa87fea27a539e9b3,
        0x97c560ba6b0919b5, 0x88b402f7fd7553ab, 0xf64335bcf065d3a0, 0xddd0467c64bce4c4, 0xc7caba6e7c5382ed,
        0xb3f4e093db73a0b7, 0xa21727db38cb0053, 0x91ff83775423cc29, 0x8380dea93da4bc82, 0xece53cec4a314f00,
        0xd5605fcdcf32e217, 0xc0314325637a1978, 0xad1c8eab5ee43ba2, 0x9becce62836ac5b0, 0x8c71dcd9ba0b495c,
        0xfd00b89747823938, 0xe3e27a444d8d991a)
_E16 = (54, 107, 160, 213, 266, 319, 373, 426, 479, 532, 585, 638, 691, 745, 798, 851, 904, 957, 1010, 1064, 1117)
_M64 = (1 << 64) - 1


def _mul32(a: int, b: int) -> int:
    return (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)


def _mul64_lossy(a: int, b: int, exp: int) -> Tuple[int, int]:
    val = (_mul32(a >> 32, b >> 32) + (_mul32(a >> 32, b) >> 32) + (_mul32(a, b >> 32) >> 32)) & _M64
    if not val & (1 << 63):
        val = (val << 1) & _M64
        exp -= 1
    return val, exp


def _mono_number_to_double(digits: str, scale: int, negative: int) -> float:
    """Port of mono's managed Number.NumberToDouble (value = 0.digits x 10^scale)."""
    total = remaining = len(digits)
    count = min(remaining, 9)
    remaining -= count
    val = int(digits[:count])
    if remaining > 0:
        count = min(remaining, 9)
        remaining -= count
        mult = (_P10[count - 1] >> (64 - _E10[count - 1])) & 0xFFFFFFFF
        val = _mul32(val, mult) + int(digits[9:9 + count])
    scale -= total - remaining
    absscale = abs(scale)
    if absscale >= 22 * 16:
        bits = 0x7FF0000000000000 if scale > 0 else 0
    else:
        exp = 64
        for mask, shift in ((0xFFFFFFFF00000000, 32), (0xFFFF000000000000, 16), (0xFF00000000000000, 8),
                            (0xF000000000000000, 4), (0xC000000000000000, 2), (0x8000000000000000, 1)):
            if val & mask == 0:
                val = (val << shift) & _M64
                exp -= shift
        index = absscale & 15
        if index:
            mexp = _E10[index - 1]
            exp += (-mexp + 1) if scale < 0 else mexp
            val, exp = _mul64_lossy(val, _P10[index + (15 if scale < 0 else 0) - 1], exp)
        index = absscale >> 4
        if index:
            mexp = _E16[index - 1]
            exp += (-mexp + 1) if scale < 0 else mexp
            val, exp = _mul64_lossy(val, _P16[index + (21 if scale < 0 else 0) - 1], exp)
        if val & (1 << 10):
            tmp = val + ((1 << 10) - 1) + ((val >> 11) & 1)
            if tmp > _M64:
                tmp = ((tmp & _M64) >> 1) | 0x8000000000000000
                exp += 1
            val = tmp
        exp += 0x3FE
        if exp <= 0:
            if exp == -52 and val >= 0x8000000000000058:
                val = 1
            elif exp <= -52:
                val = 0
            else:
                val >>= (-exp + 12)
        elif exp >= 0x7FF:
            val = 0x7FF0000000000000
        else:
            val = (exp << 52) + ((val >> 11) & 0x000FFFFFFFFFFFFF)
        bits = val
    if negative:
        bits |= 1 << 63
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _mono_g(value: float, precision: int) -> str:
    """Mono "G<precision>": digits of the EXACT binary value rounded half away
    from zero (not Python's half-even: 90619585105019.0625 -> ...063)."""
    sign, digits, adjusted = _rounded_digits(value, precision)
    body = digits.rstrip("0") or "0"
    prefix = "-" if sign else ""
    if -5 < adjusted < precision:
        if adjusted >= 0:
            whole = body[:adjusted + 1].ljust(adjusted + 1, "0")
            frac = body[adjusted + 1:]
        else:
            whole, frac = "0", "0" * (-adjusted - 1) + body
        return prefix + whole + ("." + frac if frac else "")
    mantissa = body[0] + ("." + body[1:] if len(body) > 1 else "")
    return "{}{}E{}{:02d}".format(prefix, mantissa, "-" if adjusted < 0 else "+", abs(adjusted))
