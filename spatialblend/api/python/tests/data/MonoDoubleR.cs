// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Writes "<hex bits>\t<double.ToString("R", InvariantCulture)>" per line on
// Unity's MonoBleedingEdge, for a deterministic value set:
//  - fixed edge cases, - float32 values widened to double (exact binary ties
//  at the 17th digit are common), - random 64-bit patterns (all finite
//  doubles incl. subnormals), - random doubles in [-100, 100].
using System;
using System.Globalization;
using System.IO;
using System.Text;
class Ref {
  static ulong s = 0x9E3779B97F4A7C15UL;
  static ulong Next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  static void Emit(StringBuilder b, double d) {
    if (double.IsNaN(d) || double.IsInfinity(d)) return;
    b.Append(BitConverter.DoubleToInt64Bits(d).ToString("x16")).Append('\t')
     .Append(d.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
  }
  static void Main(string[] a) {
    int n = int.Parse(a[1]);
    var b = new StringBuilder();
    double[] fixedCases = { 0.0, -0.0, 1.0, -1.0, 0.1, 1.0 / 3.0, 0.99924087524414063,
      5.0867259815001304, 3.6129, 90619585105019.0625, 2.2204460492503131E-13,
      1.1960510015487671, 1e15, 1e-5, 1e-4, 1e14, 123456789012345.6,
      double.Epsilon, -double.Epsilon, double.MaxValue, double.MinValue,
      2.2250738585072014E-308, 0.5, 0.25, 12.5 };
    foreach (var d in fixedCases) Emit(b, d);
    for (int i = 0; i < n; ++i) {
      ulong r = Next();
      Emit(b, (double)BitConverter.ToSingle(BitConverter.GetBytes((uint)r), 0));
      Emit(b, (double)(float)(((double)(r >> 11) / (1UL << 53)) * 20.0 - 10.0));
      Emit(b, BitConverter.Int64BitsToDouble((long)Next()));
      Emit(b, ((double)(Next() >> 11) / (1UL << 53)) * 200.0 - 100.0);
    }
    File.WriteAllText(a[0], b.ToString(), new UTF8Encoding(false));
  }
}
