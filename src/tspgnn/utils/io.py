from __future__ import annotations
from pathlib import Path
import io, gzip, urllib.request
import numpy as np

def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in arrays.items():
        if isinstance(v, (str, bytes)): v = np.array(v)
        out[k] = v
    np.savez_compressed(path, **out)

def load_npz(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}

def urlretrieve(url: str, timeout: int = 30) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None

def download_tsplib_file(name: str, raw_dir: Path, kind: str) -> Path | None:
    assert kind in ("tsp","opt.tour")
    dst = raw_dir / f"{name}.{kind}"
    if dst.exists(): return dst
    base_h="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    base_h_http="http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    base_zib="http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/"
    urls=[f"{base_h}{name}.{kind}", f"{base_h}{name}.{kind}.gz",
          f"{base_h_http}{name}.{kind}", f"{base_h_http}{name}.{kind}.gz",
          f"{base_zib}{name}.{kind}", f"{base_zib}{name}.{kind}.gz"]
    for u in urls:
        data = urlretrieve(u)
        if data is None: continue
        try:
            text = gzip.GzipFile(fileobj=io.BytesIO(data)).read() if u.endswith(".gz") else data
            dst.write_bytes(text)
            return dst
        except Exception:
            continue
    return None
