import numpy as np
from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt
from juliacall import Main as jl
from diffeqpy import de

@dataclass
class SanoSawada:
    ts: np.ndarray
    num_neighbor: int
    dt: float
    step_jac: int # step_jac*dt の刻み幅でヤコビ行列を計算
    jac_len: int # 接ダイナミクスの総ステップ数+1
    epsilon: float
    td_idx: np.ndarray
    sampling_neighbors: str = "sequential"
    model: Any = None

    def __post_init__(self):
        self.l_end = np.abs(np.min(self.td_idx)) # ts のうちヤコビ行列を評価に用いられる左端の添字
        if self.jac_len > (self.ts.shape[0]-self.step_jac-self.l_end+1)//self.step_jac:
            raise ValueError("jac_len shoud be smaller")
        self.r_end = (self.jac_len-1)*self.step_jac + self.l_end

        acceptable_sampling_methods = ["sequential", "ascending", "random"]
        if self.sampling_neighbors not in acceptable_sampling_methods:
            raise ValueError(f"{self.sampling_neighbors} is not an acceptable value")
        
        de.seval("using LinearAlgebra") # julia の QR 分解を使えるようにしておく
        # Benettin-Shimada-Nagashima step
        self.bns = de.seval(""" 
        function bns(du,u,p,t)
            N = size(p,1)
            Q, R = qr(p[:,:,t] * reshape(u[1:N^2], N, N))
            du[1:N^2] = vec(Q)
            du[N^2+1:end] = log.(abs.(diag(R)))
        end""")

    def set_radius_of_ball_for_neighboring_points_search(self):
        self.radius = self.epsilon*(np.max(self.ts)-np.min(self.ts))

    def jacobian_reconstruction(self):
        ts_recon = np.empty((self.ts.shape[0],self.td_idx.shape[0])) # 時間遅れ座標で再構成した時系列を格納
        for i in range(ts_recon.shape[1]): 
             ts_recon[:,i] = np.roll(self.ts,-self.td_idx[i])
        
        diffs = ts_recon[np.newaxis, :, :] - ts_recon[self.l_end:self.r_end+self.step_jac+1:self.step_jac, np.newaxis, :] # 再構成時系列の差
        dist = np.linalg.norm(diffs[:self.jac_len], axis=2)
        mask = (dist < self.radius)[:,:-self.step_jac] # 半径 self.radius の近傍に属する添字に True を格納する配列
        self_idx = np.arange(self.l_end,self.r_end+1,self.step_jac)
        mask[np.arange(mask.shape[0]),self_idx] = False # 自分自身を近傍から除外

        diffs_neighbors = np.empty((self.jac_len, self.num_neighbor, len(self.td_idx))) # 近傍に属する接ベクトルを格納する配列
        diffs_neighbors_f = np.empty_like(diffs_neighbors) # 上の接ベクトルを step_jac*dt だけ発展させたものを格納

        for i in range(self.jac_len):
            indices = np.where(mask[i])[0]
            if len(indices) < self.num_neighbor:
                raise ValueError("Insufficient number of neighboring points.")
            else: 
                if self.sampling_neighbors == "sequential": 
                    indices = indices[:self.num_neighbor]
                elif self.sampling_neighbors == "ascending": 
                    sorted_indices = indices[np.argsort(dist[i, indices])]
                    indices = sorted_indices[:self.num_neighbor]
                elif self.sampling_neighbors == "random":
                    indices = np.random.choice(indices, self.num_neighbor, replace=False)
                else:
                    raise ValueError(f"{self.sampling_neighbors} is not an acceptable value")

            diffs_neighbors[i,:,:] = diffs[i,indices,:]
            diffs_neighbors_f[i,:,:] = diffs[i+1,indices+self.step_jac,:]

        self.jac = np.empty((self.td_idx.shape[0],self.td_idx.shape[0],self.jac_len),order='F') # ヤコビ行列を格納．juliaに渡すので列優先にしておく
        for i in range(self.jac_len):
            J_transpose, _, rank, _ = np.linalg.lstsq(diffs_neighbors[i],diffs_neighbors_f[i],rcond=None)
            if rank != self.jac.shape[0]:
                print("Rank deficiency at idx {}".format(i))
            self.jac[:,:,i] = J_transpose.T

    def bns_steps(self):
        tspan = (0, self.jac_len)
        u0 = np.zeros(self.td_idx.shape[0]**2+self.td_idx.shape[0]) # 0:self.td_idx.shape[0]**2 にBNSアルゴリズムのPを，残りにRの対角成分の対数を格納
        u0[np.arange(0,len(u0),self.td_idx.shape[0]+1)]=1.0 # BNSアルゴリズムの初期条件を単位行列とする
        prob = de.DiscreteProblem(self.bns, u0, tspan, self.jac)
        sol = de.solve(prob)
        self.R = np.stack([u_i[self.td_idx.shape[0]**2:] for u_i in sol.u[1:]]) # Rの対角成分の対数のみ保持．第一行は人工的なゼロが入っているので除外

    def estimate_lyapunov_spectrum(self,transient=0,length=None):
        if length is None:
            length = self.jac_len-transient
        self.LS = np.mean(self.R[transient:length+transient,:],axis=0)/self.step_jac/self.dt
        print("Estimated Lyapunov spectrum:", self.LS)
