import json, numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
import tools.encode_getup_gate as g
def main():
    torch.set_grad_enabled(False)
    d=np.load('runs/diagnostics/gate-probes.npz');pr=tuple(d[f'arr_{i}'] for i in range(4))
    for bump in (False,True):
        for skip in (1,2,3):
            path=f'runs/diagnostics/gate-{"bump" if bump else "step"}{skip}.zip'
            g.encode('runs/diagnostics/getup-forward-20261103/getup.zip',path,(0,-0.05,0.85,1.05,1),rank1=skip,probes=pr,bump=bump)
            with ProcessPoolExecutor(6,initializer=g.init,initargs=(path,'encoded')) as ex:
                rows=list(ex.map(g.rollout,[(None,*s) for s in g.STARTS]))
            json.dump(rows,open(path+'.json','w'))
            print(json.dumps(dict(path=path,success=sum(r['success'] for r in rows),standard=[r['success'] for r in rows[:3]],first=[r['first_stable'] for r in rows[:3]])),flush=True)
if __name__=='__main__': main()
