"""Compare two quantized-mesh tile trees produced on different platforms.

Byte equality is too strict across platforms: the tile header stores
positions computed with sin and cos, and the C maths libraries of different
platforms may differ in the last bit. So each tile is compared in two parts:

  - the mesh (vertices, triangles, edge lists): must be identical;
  - the header (centre, height range, bounding sphere, horizon point):
    must agree within a small relative tolerance.

Standard library only. Usage: compare_tiles.py <tiles-a> <tiles-b>
"""
import gzip
import pathlib
import struct
import sys

HEADER = struct.Struct("<3d2f4d3d")
HEADER_NAMES = ["centerX", "centerY", "centerZ", "minimumHeight", "maximumHeight",
                "sphereX", "sphereY", "sphereZ", "sphereRadius",
                "horizonX", "horizonY", "horizonZ"]
RELATIVE_TOLERANCE = 1e-9
FLOAT_TOLERANCE = 1e-6          # the two heights are 32-bit floats


def tiles(root):
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*.terrain")}


def payload(path):
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def relativeDifference(a, b):
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale


def unzigzagDelta(values):
    decoded, value = [], 0
    for raw in values:
        value += (raw >> 1) ^ -(raw & 1)
        decoded.append(value)
    return decoded


def vertices(data):
    count = struct.unpack_from("<I", data, HEADER.size)[0]
    offset = HEADER.size + 4
    arrays = []
    for _ in range(3):
        arrays.append(unzigzagDelta(struct.unpack_from("<%dH" % count, data, offset)))
        offset += 2 * count
    return arrays


def describeMeshDifference(a, b):
    ua, va, ha = vertices(a)
    ub, vb, hb = vertices(b)
    if len(ua) != len(ub):
        return "vertex count %d vs %d" % (len(ua), len(ub))
    if ua != ub or va != vb:
        return "same vertex count, different vertex positions"
    minimum, maximum = HEADER.unpack_from(a)[3:5]
    step = (maximum - minimum) / 32767.0
    worst = max(abs(x - y) for x, y in zip(ha, hb))
    if worst:
        return "same positions, heights differ by up to %d steps (%.4f m)" % (worst, worst * step)
    return "same vertices, different triangles or edge lists"


def main(rootA, rootB):
    a, b = tiles(pathlib.Path(rootA)), tiles(pathlib.Path(rootB))
    print("tiles: %d vs %d" % (len(a), len(b)))
    failures = []
    if not a or not b:
        failures.append("no tiles to compare")
    if set(a) != set(b):
        failures.append("different tile sets: only in first %s, only in second %s"
                        % (sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]))

    identicalBytes = identicalMesh = 0
    worstHeader = (0.0, None, None)
    for name in sorted(set(a) & set(b)):
        dataA, dataB = payload(a[name]), payload(b[name])
        if dataA == dataB:
            identicalBytes += 1
            identicalMesh += 1
            continue
        if dataA[HEADER.size:] == dataB[HEADER.size:]:
            identicalMesh += 1
        else:
            failures.append("%s: mesh differs: %s" % (name, describeMeshDifference(dataA, dataB)))
        for index, (x, y) in enumerate(zip(HEADER.unpack_from(dataA), HEADER.unpack_from(dataB))):
            difference = relativeDifference(x, y)
            limit = FLOAT_TOLERANCE if index in (3, 4) else RELATIVE_TOLERANCE
            if difference > worstHeader[0]:
                worstHeader = (difference, name, HEADER_NAMES[index])
            if difference > limit:
                failures.append("%s: header %s differs: %r vs %r" % (name, HEADER_NAMES[index], x, y))

    shared = len(set(a) & set(b))
    print("byte-identical tiles: %d of %d" % (identicalBytes, shared))
    print("identical meshes:     %d of %d" % (identicalMesh, shared))
    if worstHeader[1]:
        print("largest header difference: %.3g (relative) in %s, %s" % worstHeader)
    for failure in failures[:50]:
        print("FAIL " + failure)
    if failures:
        print("%d problem(s)" % len(failures))
        return 1
    print("Equivalent tiles on both platforms.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
