"""Shared helpers for the E04 builders (pre-registration: vlm-spatial-probe experiments/E04-more-measures/PREREG.md;
output format: built_e04/SCHEMA.md). Remote zips are read by HTTP byte range; nothing but the 640-px JPEGs and the
row JSON is written."""
import io
import random
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent                                   # built_e04/
SEED = 2604
MAX_SIDE = 640


def _get(url, a=None, b=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'} if a is not None else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = r.read()
                if a is not None and r.status != 206:
                    raise IOError(f'no range support: {r.status}')
                return d
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests (enough for zipfile's central directory)."""
    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])

    def _fetch(self, i):
        if i not in self.cache:
            a = i * self.block
            if len(self.cache) > 64:
                self.cache = {}
            self.cache[i] = _get(self.url, a, min(self.size, a + self.block) - 1)
        return self.cache[i]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            i = p // self.block
            blk = self._fetch(i)
            s = p - i * self.block
            take = blk[s:s + n - len(out)]
            out += take
            p += len(take)
        buf[:n] = out
        self.pos += n
        return n


class RemoteZip:
    """Central directory read once by range; each member then fetched with one ranged GET (thread-safe)."""
    def __init__(self, url, block=1 << 20):
        self.url = url
        self.zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url, block), block))
        self.info = {i.filename: i for i in self.zf.infolist()}

    def namelist(self):
        return list(self.info)

    def read(self, name):
        i = self.info[name]
        b = _get(self.url, i.header_offset, i.header_offset + 30 + len(i.filename.encode()) + 2048 + i.compress_size)
        assert b[:4] == b'PK\x03\x04', name
        n, m = struct.unpack('<HH', b[26:30])
        data = b[30 + n + m: 30 + n + m + i.compress_size]
        if i.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)
        elif i.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f'compression {i.compress_type}')
        if zlib.crc32(data) != i.CRC:
            raise IOError(f'CRC mismatch {name}')
        return data


def save_image(img, path):
    """As vlm-spatial-probe src/build_dataset.py save_image: RGB JPEG q93, long side at most 640 px."""
    from PIL import Image
    w, h = img.size
    s = min(1.0, MAX_SIDE / max(w, h))
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(path, quality=93)
    return list(img.size)


def split_by_group(rows, test_share=0.4, seed=SEED):
    """60/40 by group: groups sorted, shuffled with the seed, then each goes to test while that brings the test share of
    rows closer to 40% (whole groups only)."""
    groups = sorted({r['group'] for r in rows})
    random.Random(seed).shuffle(groups)
    size = {g: sum(r['group'] == g for r in rows) for g in groups}
    total, test, n_test = len(rows), set(), 0
    for g in groups:
        if abs(n_test + size[g] - test_share * total) < abs(n_test - test_share * total):
            test.add(g)
            n_test += size[g]
    for r in rows:
        r['split'] = 'test' if r['group'] in test else 'train'
    return rows


def facing_deg(f, v, up):
    """SCHEMA angle: f the forward (or motion) direction, v the unit vector from the object to the camera, both projected
    onto the ground plane (normal up); 0 = towards the camera, 90 = image right, 180 = away, 270 = image left."""
    import numpy as np
    up = np.asarray(up, float) / np.linalg.norm(up)
    f = np.asarray(f, float) - np.dot(f, up) * up
    v = np.asarray(v, float) - np.dot(v, up) * up
    return float(np.degrees(np.arctan2(np.dot(np.cross(v, f), up), np.dot(v, f))) % 360)
