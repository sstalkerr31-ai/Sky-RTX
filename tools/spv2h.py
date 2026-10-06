#!/usr/bin/env python3
"""Convert a compiled SPIR-V file into the C++ header the layer embeds.
   glslangValidator -V --target-env vulkan1.2 layer/rt_fx.comp -o rtfx.spv
   python tools/spv2h.py rtfx.spv layer/rtfx_spv.h kRtFxSpv rt_fx.comp
"""
import struct, sys
src, dst, name, comp = sys.argv[1:5]
b = open(src, "rb").read()
w = struct.unpack("<%dI" % (len(b) // 4), b)
out = ["// generated from %s with glslang -V --target-env vulkan1.2 (SPIR-V, %d words)\n" % (comp, len(w)),
       "#pragma once\n", "#include <cstdint>\n", "static const uint32_t %s[] = {\n" % name]
for i in range(0, len(w), 8):
    out.append("    " + ", ".join("0x%08x" % x for x in w[i:i + 8]) + ",\n")
out.append("};\n")
open(dst, "w").write("".join(out))
print("wrote", dst, len(w), "words")
