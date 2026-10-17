import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { expect, test } from 'vitest';
import { openNpy, readAt, readBlock } from './npy';

// needs python+numpy (same one CI installs). Skips quietly if absent.
const py = process.env.PYTHON ?? 'python3';
function haveNumpy() {
  try { execFileSync(py, ['-c', 'import numpy']); return true; } catch { return false; }
}

test.skipIf(!haveNumpy())('reads slices written by numpy', () => {
  const f = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'npy-')), 'a.npy');
  execFileSync(py, ['-c', `import numpy as np;a=np.arange(2*3*4,dtype=np.float32).reshape(2,3,4);a[1,2,3]=np.nan;np.save(${JSON.stringify(f)},a)`]);
  const a = openNpy(f);
  expect(a.shape).toEqual([2, 3, 4]);
  expect(Array.from(readBlock(a, [1, 0]))).toEqual([12, 13, 14, 15]);
  expect(readAt(a, [0, 2, 1])).toBe(9);
  expect(Number.isNaN(readAt(a, [1, 2, 3]))).toBe(true);
  expect(() => readBlock(a, [2])).toThrow(RangeError);
});
