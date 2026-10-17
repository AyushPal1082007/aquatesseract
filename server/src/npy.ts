// Minimal .npy reader. Reads slices straight from disk (no full load) = poor man's memmap.
import fs from 'node:fs';

export interface Npy {
  path: string;
  shape: number[];
  dtype: string;
  itemSize: number;
  dataOffset: number;
  fd: number;
}

const SIZES: Record<string, number> = { '<f4': 4, '<f8': 8, '|u1': 1, '|b1': 1, '|i1': 1, '<i4': 4, '<u2': 2 };
const cache = new Map<string, Npy>();

export function openNpy(path: string): Npy {
  const hit = cache.get(path);
  if (hit) return hit;
  const fd = fs.openSync(path, 'r');
  const pre = Buffer.alloc(12);
  fs.readSync(fd, pre, 0, 12, 0);
  if (pre[0] !== 0x93 || pre.toString('latin1', 1, 6) !== 'NUMPY') throw new Error(`not an npy file: ${path}`);
  const major = pre[6];
  const hlen = major === 1 ? pre.readUInt16LE(8) : pre.readUInt32LE(8);
  const hstart = major === 1 ? 10 : 12;
  const hb = Buffer.alloc(hlen);
  fs.readSync(fd, hb, 0, hlen, hstart);
  const h = hb.toString('latin1');
  const dtype = /'descr':\s*'([^']+)'/.exec(h)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(h)?.[1];
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(h)?.[1];
  if (!dtype || !fortran || shapeStr === undefined) throw new Error(`bad npy header: ${path}`);
  if (fortran === 'True') throw new Error(`fortran_order not supported: ${path}`);
  const itemSize = SIZES[dtype];
  if (!itemSize) throw new Error(`unsupported dtype ${dtype}: ${path}`);
  const shape = shapeStr.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const info = { path, shape, dtype, itemSize, dataOffset: hstart + hlen, fd };
  cache.set(path, info);
  return info;
}

function decode(buf: Buffer, dtype: string, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    switch (dtype) {
      case '<f4': out[i] = buf.readFloatLE(i * 4); break;
      case '<f8': out[i] = buf.readDoubleLE(i * 8); break;
      case '<i4': out[i] = buf.readInt32LE(i * 4); break;
      case '<u2': out[i] = buf.readUInt16LE(i * 2); break;
      case '|i1': out[i] = buf.readInt8(i); break;
      default: out[i] = buf[i];
    }
  }
  return out;
}

const prod = (a: number[]) => a.reduce((x, y) => x * y, 1);

/** Contiguous block: fix leading indices, return all remaining dims flattened (C order). */
export function readBlock(a: Npy, lead: number[]): Float64Array {
  const rest = a.shape.slice(lead.length);
  const count = prod(rest);
  let start = 0;
  lead.forEach((ix, k) => {
    if (ix < 0 || ix >= a.shape[k]) throw new RangeError(`index ${ix} out of range for axis ${k} (size ${a.shape[k]})`);
    start += ix * prod(a.shape.slice(k + 1));
  });
  const buf = Buffer.alloc(count * a.itemSize);
  fs.readSync(a.fd, buf, 0, buf.length, a.dataOffset + start * a.itemSize);
  return decode(buf, a.dtype, count);
}

/** Single element at full index. */
export function readAt(a: Npy, idx: number[]): number {
  return readBlock(a, idx)[0];
}
