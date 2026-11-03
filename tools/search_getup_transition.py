"""Black-box forward simulation of dynamic transitions from a measured sit."""
import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
from tools.design_getup_forward import HumanoidSim, joint_pose, fast_step, metrics
from tools.build_manual_getup import evaluate, latent_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--population',type=int,default=64)
    p.add_argument('--output',default='artifacts/manual_getup/transition.npz')
    args=p.parse_args()
    data=np.load('artifacts/manual_getup/sit_cem.npz')
    initial,q0=data['initial'],data['initial_target']
    s=HumanoidSim(physics_dt=.001,action_mapping='joint_limits')
    base=np.load('artifacts/manual_getup/manual_parameters.npz')['parameters']
    evaluate(s,latent_table(s,base[:49].reshape(7,7)),base[49],base[50],steps=160)
    initial=s.data.qpos[:27].copy();initial_velocity=s.data.qvel[:26].copy()
    rng=np.random.default_rng(111)
    poses=np.array([[.8,-1.5,2.3,-.8,-1.3,1.8,.4],
                    [.8,-1.6,2.3,-.8,.8,.1,.3],
                    [.3,-1.2,2.1,-.8,-.4,.5,.3]])
    mean=np.r_[poses.ravel(),[.5,.5,1.5]]
    low=np.r_[np.tile([-.5,-1.6,0,-.8,-2.4,0,.08],3),[.12]*3]
    high=np.r_[np.tile([.8,1.1,2.3,.8,1.2,2.3,1.5],3),[2.5]*3]
    std=np.r_[np.tile([.4,.8,.8,.5,1.2,1.,.5],3),[.4]*3]
    out=Path(args.output)
    if out.exists():mean=np.load(out)['parameters'].copy()
    def run(v,keep=False):
        s.reset('supine');s.data.qpos[:27]=initial;s.data.qvel[:26]=initial_velocity;mujoco.mj_forward(s.model,s.data)
        prev=q0.copy();rows=[];qposes=[];targets=[];velocities=[]
        qs=[joint_pose(a,s) for a in v[:21].reshape(3,7)]+[s.default_q]
        qs=[np.clip(q,s.default_q-.964*s.action_scale_low,s.default_q+.964*s.action_scale_high) for q in qs]
        for q,dur in zip(qs,[*v[21:],1.]):
            n=max(1,round(dur*50))
            for k in range(n):
                a=(k+1)/n;a=a*a*(3-2*a);target=prev*(1-a)+q*a;fast_step(s,target);rows.append(metrics(s))
                if keep:qposes.append(s.data.qpos[:27].copy());targets.append(target);velocities.append(s.data.qvel[:26].copy())
            prev=q.copy()
        for _ in range(100):
            fast_step(s,prev);rows.append(metrics(s))
            if keep:qposes.append(s.data.qpos[:27].copy());targets.append(prev.copy());velocities.append(s.data.qvel[:26].copy())
        rows=np.array(rows);qual=np.minimum(np.clip(rows[:,0]/.94,0,1),np.clip(rows[:,1]/1.5,0,1))*np.maximum(0,np.minimum(rows[:,2],rows[:,3]))
        score=4*np.mean(qual[-75:])+2*np.max(qual)+np.mean(qual)-.02*np.mean(np.minimum(rows[-75:,4]+rows[-75:,5],10))
        return score,rows,qposes,targets,velocities
    best=mean.copy();best_score=run(best)[0]
    for it in range(args.iterations):
        cs=np.clip(mean+rng.normal(size=(args.population,len(mean)))*std,low,high);cs[0]=best
        scores=np.array([run(c)[0] for c in cs]);order=np.argsort(scores)[::-1];elite=cs[order[:max(6,args.population//6)]]
        mean=.2*mean+.8*elite.mean(0);std=np.maximum(.04,.45*std+.55*elite.std(0))
        if scores[order[0]]>best_score:best_score=float(scores[order[0]]);best=cs[order[0]].copy()
        score,rows,qp,targets,velocities=run(best,keep=True)
        np.savez(out,parameters=best,score=best_score,metrics=rows,qpos=qp,qvel=velocities,targets=targets,initial=initial,initial_target=q0)
        print(json.dumps(dict(iteration=it,best=best_score,max_height=float(rows[:,0].max()),tail=rows[-75:].mean(0).tolist())),flush=True)


if __name__=='__main__':main()
