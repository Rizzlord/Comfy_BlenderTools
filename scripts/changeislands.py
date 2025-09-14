import argparse, numpy as np, trimesh, pathlib, sys, math

def tri_area2d(a,b,c):
    return abs(0.5*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])))

def build_uv_islands(F, uv):
    e2f = {}
    for fi,(a,b,c) in enumerate(F):
        for e in [(a,b),(b,c),(c,a)]:
            x,y = (e[0],e[1]) if e[0]<=e[1] else (e[1],e[0])
            e2f.setdefault((x,y), []).append(fi)
    adj = [[] for _ in range(F.shape[0])]
    for faces in e2f.values():
        if len(faces)==2:
            u,v=faces
            if np.allclose(uv[F[u,0]],uv[F[v,0]]) and np.allclose(uv[F[u,1]],uv[F[v,1]]): pass
            adj[u].append(v); adj[v].append(u)
    comp = np.full(F.shape[0], -1, dtype=np.int64)
    cid=0
    for i in range(F.shape[0]):
        if comp[i]!=-1: continue
        st=[i]; comp[i]=cid
        while st:
            u=st.pop()
            for v in adj[u]:
                if comp[v]==-1:
                    comp[v]=cid; st.append(v)
        cid+=1
    islands=[[] for _ in range(cid)]
    for i,c in enumerate(comp): islands[c].append(i)
    return islands, comp

def island_uv_indices(is_faces, F):
    s=set()
    for fi in is_faces:
        a,b,c = F[fi]
        s.add(a); s.add(b); s.add(c)
    return np.array(sorted(list(s)), dtype=np.int64)

def island_area_uv(is_faces, F, uv):
    A=0.0
    for fi in is_faces:
        a,b,c = F[fi]
        A+=tri_area2d(uv[a],uv[b],uv[c])
    return A

def face_neighbors_by_3d(F):
    mp={}
    for fi,(a,b,c) in enumerate(F):
        for e in [(a,b),(b,c),(c,a)]:
            x,y=(e[0],e[1]) if e[0]<=e[1] else (e[1],e[0])
            mp.setdefault((x,y), []).append(fi)
    adj=[set() for _ in range(F.shape[0])]
    for faces in mp.values():
        if len(faces)==2:
            u,v=faces
            adj[u].add(v); adj[v].add(u)
    return [list(s) for s in adj], mp

def island_neighbors_by_3d(islands, face_adj, face_to_island):
    neigh=[set() for _ in range(len(islands))]
    for i,faces in enumerate(islands):
        for fi in faces:
            for nb in face_adj[fi]:
                j=face_to_island[nb]
                if i!=j: neigh[i].add(j)
    return [list(s) for s in neigh]

def extract_shared_3d_edges_between_islands(iA,iB,islands,F,edge_map):
    facesA=set(islands[iA]); facesB=set(islands[iB])
    shared=[]
    for (e,faces) in edge_map.items():
        if len(faces)==2:
            u,v=faces
            inA = (u in facesA) or (v in facesA)
            inB = (u in facesB) or (v in facesB)
            if inA and inB:
                shared.append(e)
    return shared

def face_uv_edge(F, uv, fi, va, vb):
    fa,fb,fc = F[fi]
    if (fa==va and fb==vb) or (fb==va and fa==vb):
        return np.array([uv[fa],uv[fb]]), (fa,fb)
    if (fb==va and fc==vb) or (fc==va and fb==vb):
        return np.array([uv[fb],uv[fc]]), (fb,fc)
    if (fc==va and fa==vb) or (fa==va and fc==vb):
        return np.array([uv[fc],uv[fa]]), (fc,fa)
    return None

def best_similarity_transform_2d(src, dst):
    c1 = src.mean(axis=0); c2 = dst.mean(axis=0)
    X = src - c1; Y = dst - c2
    sx = np.sqrt((X**2).sum()/src.shape[0]); sy = np.sqrt((Y**2).sum()/dst.shape[0])
    if sx < 1e-12: return 1.0, np.eye(2), (c2 - c1)
    Xn = X/sx; Yn = Y/sy
    H = Xn.T @ Yn
    U,S,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1,:]*=-1; R = Vt.T @ U.T
    s = S.sum()
    scale = (sy/sx)*s
    t = c2 - (scale*(R@c1))
    return scale, R, t

def apply_similarity(UV, idxs, s, R, t):
    UV[idxs] = (UV[idxs] @ R.T) * s + t

def weld_uvs_along_pairs(UV, pairs, eps):
    for (a,b) in pairs:
        if np.linalg.norm(UV[a]-UV[b])<=eps:
            m=(UV[a]+UV[b])*0.5
            UV[a]=m; UV[b]=m

def try_merge_island(A,B,islands,F,UV,edge_map,eps_align,face_to_island):
    shared_edges = extract_shared_3d_edges_between_islands(A,B,islands,F,edge_map)
    if not shared_edges: return False
    uv_pairs=[]; src_pts=[]; dst_pts=[]
    for (va,vb) in shared_edges:
        faces = edge_map[(min(va,vb),max(va,vb))]
        if len(faces)!=2: continue
        fa, fb = faces
        ia = face_to_island[fa]; ib = face_to_island[fb]
        fiA = fa if ia==A else (fb if ib==A else None)
        fiB = fb if ia==A else (fa if ib==B else None)
        if fiA is None or fiB is None: continue
        eA = face_uv_edge(F, UV, fiA, va, vb)
        eB = face_uv_edge(F, UV, fiB, vb, va)
        if eA is None or eB is None:
            eA = face_uv_edge(F, UV, fiA, vb, va)
            eB = face_uv_edge(F, UV, fiB, va, vb)
        if eA is None or eB is None: continue
        (uvA,(ta,tb)) = eA; (uvB,(tc,td)) = eB
        src_pts.append(uvA[0]); src_pts.append(uvA[1])
        dst_pts.append(uvB[0]); dst_pts.append(uvB[1])
        uv_pairs.append((ta,tc)); uv_pairs.append((tb,td))
    if len(src_pts)<2: return False
    src = np.array(src_pts); dst = np.array(dst_pts)
    s,R,t = best_similarity_transform_2d(src, dst)
    idxs = island_uv_indices(islands[A], F)
    apply_similarity(UV, idxs, s, R, t)
    weld_uvs_along_pairs(UV, uv_pairs, eps_align)
    islands[B].extend(islands[A]); islands[A].clear()
    for fi in range(F.shape[0]):
        if face_to_island[fi]==A: face_to_island[fi]=B
    return True

def pack_islands(islands, F, UV, margin):
    bbs=[]; idxs=[]
    for faces in islands:
        if not faces:
            bbs.append((0,0,0,0)); idxs.append(np.array([],dtype=np.int64)); continue
        uv_idx = island_uv_indices(faces, F)
        P = UV[uv_idx]
        mn = P.min(axis=0); mx = P.max(axis=0)
        bbs.append((mn[0],mn[1],mx[0],mx[1])); idxs.append(uv_idx)
    sizes=[]
    for (x0,y0,x1,y1) in bbs:
        w=max(1e-9,x1-x0); h=max(1e-9,y1-y0); sizes.append((w,h))
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1], reverse=True)
    cursor_x=margin; cursor_y=margin; row_h=0.0; atlas_w=1.0; placements=[None]*len(sizes)
    for i in order:
        w,h = sizes[i]
        if cursor_x + w + margin > atlas_w:
            cursor_x = margin
            cursor_y += row_h + margin
            row_h = 0.0
        placements[i]=(cursor_x, cursor_y)
        cursor_x += w + margin
        row_h = max(row_h, h)
    max_x = 0.0; max_y = 0.0
    for i,(x,y) in enumerate(placements):
        w,h = sizes[i]
        max_x = max(max_x, x+w+margin); max_y = max(max_y, y+h+margin)
    scale = 1.0/max(max_x, max_y)
    for i,faces in enumerate(islands):
        if not faces: continue
        uv_idx = idxs[i]
        (x0,y0,x1,y1)=bbs[i]
        P = UV[uv_idx]
        P = P - np.array([x0,y0])
        P = P + np.array(placements[i])
        P = P * scale
        UV[uv_idx]=P

def process_mesh(mesh, min_uv_area, eps_align, margin):
    if not hasattr(mesh, "visual") or mesh.visual is None or mesh.visual.uv is None or len(mesh.visual.uv)==0:
        return False
    if mesh.faces is None or len(mesh.faces)==0: return False
    uv = mesh.visual.uv.astype(np.float64).copy()
    F = mesh.faces.view(np.ndarray).astype(np.int64).copy()
    islands, comp = build_uv_islands(F, uv)
    face_to_island = np.array(comp, dtype=np.int64)
    face_adj, edge_map = face_neighbors_by_3d(F)
    changed=True
    while changed:
        changed=False
        areas=[island_area_uv(faces, F, uv) for faces in islands]
        neigh = island_neighbors_by_3d(islands, face_adj, face_to_island)
        small = [i for i,a in enumerate(areas) if a<min_uv_area and len(islands[i])>0]
        for i in small:
            if not neigh[i]: continue
            nb = max(neigh[i], key=lambda j: areas[j] if j<len(areas) else 0.0)
            if i==nb: continue
            ok = try_merge_island(i, nb, islands, F, uv, edge_map, eps_align, face_to_island)
            if ok: changed=True
        islands = [faces for faces in islands if len(faces)>0]
        new_map = {}
        for new_i, faces in enumerate(islands):
            for fi in faces: new_map[fi]=new_i
        for fi in range(len(face_to_island)):
            if fi in new_map: face_to_island[fi]=new_map[fi]
    pack_islands(islands, F, uv, margin)
    uv = np.clip(uv,0.0,1.0)
    mesh.visual.uv = uv.astype(np.float32)
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--min_uv_area", type=float, default=5e-4)
    p.add_argument("--eps_align", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=0.01)
    args=p.parse_args()
    inp = args.input
    outp = args.output
    scene = trimesh.load(inp, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    changed_any=False
    for name, geom in list(scene.geometry.items()):
        mesh = geom
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if not mesh.is_watertight:
            mesh = mesh.copy()
        ok = process_mesh(mesh, args.min_uv_area, args.eps_align, args.margin)
        if ok: changed_any=True
        scene.geometry[name]=mesh
    data = scene.export(file_type="glb")
    pathlib.Path(outp).write_bytes(data)

if __name__=="__main__":
    main()
