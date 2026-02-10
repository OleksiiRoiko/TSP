from __future__ import annotations
import numpy as np
import math

def verify_tour(tour: np.ndarray | None, n: int) -> bool:
    if tour is None: return False
    t = np.asarray(tour)
    return t.ndim==1 and t.size==n and np.array_equal(np.sort(t), np.arange(n))

def tour_edges_undirected(tour: np.ndarray) -> np.ndarray:
    t = np.asarray(tour, dtype=np.int64); n = t.shape[0]
    E=[]
    for i in range(n):
        a,b = int(t[i]), int(t[(i+1)%n])
        if a>b: a,b=b,a
        E.append((a,b))
    return np.asarray(E, dtype=np.int64)

def tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    C = np.asarray(coords, dtype=np.float32)
    t = np.asarray(tour, dtype=np.int64); n = t.shape[0]
    L=0.0
    for i in range(n):
        a,b=t[i],t[(i+1)%n]
        L += float(np.linalg.norm(C[a]-C[b]))
    return L

def greedy_cycle_from_edges(n: int, E: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    parent = np.arange(n); rank=np.zeros(n, dtype=np.int64); deg=np.zeros(n, dtype=np.int64)
    chosen=[]
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb = find(a),find(b)
        if ra==rb: return False
        if rank[ra]<rank[rb]: parent[ra]=rb
        elif rank[ra]>rank[rb]: parent[rb]=ra
        else: parent[rb]=ra; rank[ra]+=1
        return True
    added=0
    for idx in order:
        a,b=int(E[idx,0]), int(E[idx,1])
        if deg[a]>=2 or deg[b]>=2: continue
        if find(a)==find(b) and added<n-1: continue
        chosen.append((a,b)); deg[a]+=1; deg[b]+=1; union(a,b); added+=1
        if added==n: break
    nbrs=[[] for _ in range(n)]
    for a,b in chosen: nbrs[a].append(b); nbrs[b].append(a)
    tour=[0]; prev=-1; cur=0
    for _ in range(n-1):
        u = nbrs[cur][0]
        v = nbrs[cur][1] if len(nbrs[cur])>1 else -1
        nxt = u if u!=prev else v
        prev,cur = cur,nxt
        tour.append(cur)
    return np.asarray(tour, dtype=np.int64)

def two_opt(coords: np.ndarray, tour: np.ndarray, max_passes: int=20) -> np.ndarray:
    C = np.asarray(coords, dtype=np.float32)
    T = np.asarray(tour, dtype=np.int64).copy(); n=T.shape[0]
    def seg(a,b): return float(np.linalg.norm(C[a]-C[b]))
    improved=True; passes=0
    while improved and passes<max_passes:
        improved=False; passes+=1
        for i in range(n-1):
            for j in range(i+2, n if i>0 else n-1):
                a,b=T[i],T[(i+1)%n]; c,d=T[j],T[(j+1)%n]
                old=seg(a,b)+seg(c,d); new=seg(a,c)+seg(b,d)
                if new+1e-12<old:
                    T[i+1:j+1] = T[i+1:j+1][::-1]; improved=True; break
            if improved: break
    return T

def _pair_iter(tour: np.ndarray):
    n = len(tour)
    for i in range(n):
        a = int(tour[i]); b = int(tour[(i+1) % n])
        yield a, b

def _len_euc_2d(coords: np.ndarray, tour: np.ndarray) -> int:
    """TSPLIB EUC_2D: dij = int(sqrt(dx^2+dy^2) + 0.5)."""
    total = 0
    for a, b in _pair_iter(tour):
        dx = coords[a, 0] - coords[b, 0]
        dy = coords[a, 1] - coords[b, 1]
        total += int(math.hypot(dx, dy) + 0.5)
    return total

def _len_ceil_2d(coords: np.ndarray, tour: np.ndarray) -> int:
    """TSPLIB CEIL_2D: sum of ceiled Euclidean distances (ints)."""
    total = 0
    for a, b in _pair_iter(tour):
        dx = coords[a,0] - coords[b,0]
        dy = coords[a,1] - coords[b,1]
        total += int(math.ceil(math.hypot(dx, dy)))
    return total

def _len_att(coords: np.ndarray, tour: np.ndarray) -> int:
    """
    TSPLIB ATT (pseudo-Euclidean):
      rij = sqrt(((dx^2 + dy^2)/10.0))
      tij = round(rij)
      dij = tij if tij >= rij else tij + 1
    """
    total = 0
    for a, b in _pair_iter(tour):
        dx = coords[a,0] - coords[b,0]
        dy = coords[a,1] - coords[b,1]
        rij = math.sqrt((dx*dx + dy*dy) / 10.0)
        tij = int(round(rij))
        dij = tij if tij >= rij else tij + 1
        total += dij
    return total

def _len_geo(coords: np.ndarray, tour: np.ndarray) -> int:
    """
    TSPLIB GEO: use spherical distances with coordinates in DDD.MM.
    Formula from TSPLIB95.
    """
    def to_rad(x: float) -> float:
        deg = int(x)
        minute = x - deg
        return math.pi * (deg + 5.0 * minute / 3.0) / 180.0

    lat = np.array([to_rad(float(c[0])) for c in coords], dtype=np.float64)
    lon = np.array([to_rad(float(c[1])) for c in coords], dtype=np.float64)
    rrr = 6378.388
    total = 0
    for a, b in _pair_iter(tour):
        q1 = math.cos(lon[a] - lon[b])
        q2 = math.cos(lat[a] - lat[b])
        q3 = math.cos(lat[a] + lat[b])
        dij = int(rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        total += dij
    return total

def tour_length_tsplib(coords_orig: np.ndarray, tour: np.ndarray, metric: str) -> float:
    """
    TSPLIB-compliant length (integer-valued for common metrics).
    Supports EUC_2D, CEIL_2D, ATT. Falls back to float Euclidean if unknown.
    """
    if coords_orig is None or tour is None:
        return float("nan")
    metric = (metric or "").upper()
    if metric in ("EUC_2D", "EUC2D"):
        return float(_len_euc_2d(coords_orig, tour))
    if metric in ("CEIL_2D", "CEIL2D"):
        return float(_len_ceil_2d(coords_orig, tour))
    if metric == "ATT":
        return float(_len_att(coords_orig, tour))
    if metric == "GEO":
        return float(_len_geo(coords_orig, tour))
    # Fallback: continuous Euclidean on original coords
    return float(tour_length(coords_orig.astype(np.float32), tour))
