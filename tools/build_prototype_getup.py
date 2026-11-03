"""Explicit Gaussian prototype network; no gradients or learned initialization.

Squared observation features produce Gaussian distances through linear logits.
Layer 1 normalizes prototype evidence, layer 2 merges identical PD targets.
All 334121 parameters are explicitly assigned. Raw observations remain 384D.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from stable_baselines3.common.save_util import data_to_json
from robot_rl.manual_policy import ManualGetupPolicy, PrototypeActivation
from tools.encode_getup_route import collect, replay, save
from tools.build_manual_getup import explicit_weights


def construct(x,y,features,width,lookahead=0):
    state=explicit_weights(np.zeros((7,20)))
    for k in state:
        if k!='log_std':state[k].zero_()
    # Prototype distances in selected unmodified physical observation features.
    # Squared distance's common x^2 term cancels in the softmax.
    # Therefore linear logits implement exact Gaussian interpolation.
    points=x[:,features]
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    ids=np.unique(np.searchsorted(length,np.linspace(0,length[-1],250)))
    ids=np.r_[ids,len(x)-1]
    centers=points[ids];latent=2*np.arctanh(np.arctanh(np.clip(y[np.minimum(ids+lookahead,len(y)-1)],-.964,.964))/2)
    n=len(ids)
    w1=state['mlp_extractor.policy_net.0.weight'];b1=state['mlp_extractor.policy_net.0.bias']
    w1[:n,features]=torch.tensor(2*centers/width**2,dtype=torch.float32)
    b1[:n]=torch.tensor(-np.sum(centers**2,axis=1)/width**2,dtype=torch.float32)
    b1[n:]=-1e8
    state['mlp_extractor.policy_net.2.weight'][:n,:n]=torch.eye(n)
    state['action_net.weight'][:,:n]=torch.tensor(latent.T,dtype=torch.float32)
    return state


def save_prototype(path,state):
    _save_prototype(path,state)


ACTION_SLICE=np.arange(154,174)


def chain_ids(y,budget=256,hold_every=6,settled_from=100):
    """Every action change plus sparse samples inside holds."""
    change=np.r_[True,np.abs(np.diff(y,axis=0)).max(1)>1e-4]
    ids=list(np.flatnonzero(change))
    holds=np.flatnonzero(~change)
    last_change=np.flatnonzero(change)[-1]
    ids+=list(holds[holds<min(last_change,settled_from)][::hold_every])
    ids+=list(holds[holds>last_change][::25])
    ids=np.unique(np.r_[ids,len(y)-1])
    if len(ids)>budget:
        raise ValueError(f'{len(ids)} prototypes exceed {budget} hidden units')
    return ids


def construct_chain(x,y,width,action_weight=1.,state_weights=(3.,2.,.5,.5,.5),hold_every=6,extra=None):
    """Key = previous command and physical state; value = next command.

    The previous command is an existing physical observation (last_action), so
    the network steps through a forward-validated route without phase input.
    """
    state=explicit_weights(np.zeros((7,20)))
    for k in state:
        if k!='log_std':state[k].zero_()
    features=np.r_[ACTION_SLICE,[0,3,8,9,10]]
    weights=np.r_[np.full(20,action_weight),state_weights]
    ids=chain_ids(y,hold_every=hold_every)
    keys,values=x[ids],y[ids]
    if extra is not None:
        keys,values=np.vstack([extra[0],keys]),np.vstack([extra[1],values])
    if len(keys)>256:
        raise ValueError(f'{len(keys)} prototypes exceed 256 hidden units')
    centers=keys[:,features]*weights
    latent=2*np.arctanh(np.arctanh(np.clip(values,-.964,.964))/2)
    n=len(keys)
    w1=state['mlp_extractor.policy_net.0.weight'];b1=state['mlp_extractor.policy_net.0.bias']
    w1[:n,features]=torch.tensor(2*centers*weights/width**2,dtype=torch.float32)
    b1[:n]=torch.tensor(-np.sum(centers**2,axis=1)/width**2,dtype=torch.float32)
    b1[n:]=-1e8
    state['mlp_extractor.policy_net.2.weight'][:n,:n]=torch.eye(n)
    state['action_net.weight'][:,:n]=torch.tensor(latent.T,dtype=torch.float32)
    return state,n


def _save_prototype(path,state):
    save(path,state)
    with zipfile.ZipFile(path) as ar: metadata=json.loads(ar.read('data'))
    from stable_baselines3.common.save_util import json_to_data
    data=json_to_data(json.dumps(metadata))
    data['policy_class']=ManualGetupPolicy
    data['policy_kwargs']['actor_activations']=[PrototypeActivation,torch.nn.Identity]
    data['construction']='explicit-physical-prototypes-forward-only'
    buf=io.BytesIO();torch.save(state,buf)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as ar:
        ar.writestr('data',data_to_json(data));ar.writestr('policy.pth',buf.getvalue())


def build_final(out=Path('artifacts/manual_getup')):
    """Forward-validated prone roll + supine chain; writes getup_manual.zip."""
    torch.set_grad_enabled(False)
    route=np.load(out/'forward_route.npz');x,y=route['observations'],route['actions']
    roll=np.load(out/'roll_route.npz');rx,ry=roll['observations'],roll['actions']
    ids=np.unique(np.r_[np.arange(0,49,8),48,49,np.arange(50,99,3),90,98])
    values=ry[ids].copy();values[ids>=90]=y[0]  # settled supine hands off to chain
    state,n=construct_chain(x,y,.001,hold_every=6,extra=(rx[ids],values))
    path=out/'getup_manual.zip';_save_prototype(path,state)
    return path,n


def search():
    torch.set_grad_enabled(False);torch.set_num_threads(1);o=Path('artifacts/manual_getup');x,y=collect(o);best=-1;results=[]
    for features in [[0,3,155,159],[0,3,155,159,160,168],[0,3,155,159,160,168,5,6,7],[0,3,155,159,160,168,329,330]]:
        for width in [.02,.04,.08,.15,.3]:
            st=construct(x,y,np.array(features),width);save_prototype(o/'prototype_candidate.zip',st)
            score,rows,qp,obs,a=replay(o/'prototype_candidate.zip','supine',650)
            record=dict(features=features,width=width,score=score,tail=rows[-150:].mean(0).tolist());results.append(record);print(json.dumps(record),flush=True)
            if score>best:
                best=score;save_prototype(o/'getup_prototype.zip',st);np.savez(o/'prototype_replay.npz',metrics=rows,qpos=qp)
    (o/'prototype_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':print(*build_final())
