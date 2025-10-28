import logging
import os

import numpy as np

try:
    import cupy as cp
except Exception as e:
    raise RuntimeError(f"CuPy backend requires CuPy: {e}")


class DepthOptimization:
    _raw_module = None  # class-level cache for RawModule

    def __init__(self, config):
        self.config = config
        self._buf = {}
        self._cam_cache = {
            "indices": None,
            "h": None,
            "w": None,
            "K_ref": None,
            "R_ref": None,
            "T_ref": None,
            "srcK_cp": None,
            "srcR_cp": None,
            "srcT_cp": None,
        }
        # CuPy メモリプール（割当てオーバーヘッド削減）
        try:
            mp = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(mp.malloc)
            pmp = cp.cuda.PinnedMemoryPool()
            cp.cuda.set_pinned_memory_allocator(pmp.malloc)
            logging.info("CuPy memory pools initialized")
        except Exception as e:
            logging.debug(f"CuPy memory pool init failed: {e}")
        # 代表的なカーネルをロード（RawModule）。簡易版: 評価+checkerboard伝播（4近傍）
        self._load_kernels()
        # 軽量ウォームアップ（小さなバッファで1回だけ起動）
        try:
            self._warmup()
        except Exception as e:
            logging.debug(f"CuPy warmup skipped: {e}")

    def _get_buf(self, key, shape, dtype):
        cur = self._buf.get(key)
        if cur is not None and cur.shape == shape and cur.dtype == dtype:
            return cur
        arr = cp.empty(shape, dtype=dtype)
        self._buf[key] = arr
        return arr

    def _load_kernels(self):
        # クラスキャッシュ再利用（複数インスタンスでの再コンパイル防止）
        if DepthOptimization._raw_module is not None:
            self.mod = DepthOptimization._raw_module
            self._bind_functions()
            return
        # 事前コンパイルPTXがあれば優先ロード（環境変数 or アーキ自動選択）
        ptx_path = os.getenv("MVS_CUPY_PTX")
        if not ptx_path:
            try:
                dev = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                sm = f"sm{props['major']}{props['minor']}"
                candidate_dir = os.path.join(
                    os.path.dirname(__file__), "kernels", "cupy"
                )
                cand = os.path.join(candidate_dir, f"{sm}.ptx")
                if os.path.exists(cand):
                    ptx_path = cand
                else:
                    cand = os.path.join(candidate_dir, "kernels.ptx")
                    if os.path.exists(cand):
                        ptx_path = cand
            except Exception:
                pass
        if ptx_path and os.path.exists(ptx_path):
            self.mod = cp.RawModule(path=ptx_path)
            DepthOptimization._raw_module = self.mod
            self._bind_functions()
            return

        src = r"""
extern "C" __global__ void propagate_checker4(
    float* depth, float* cost,
    const unsigned char* mask, const int H, const int W, const int color, const int use8)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    if (((r+c)&1) != color) return;

    const int idx = r*W + c;
    float best_cost = cost[idx];
    float best_depth = depth[idx];
    if (r>0 && mask[(r-1)*W + c]){
        float nc = cost[(r-1)*W + c];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + c]; }
    }
    if (r+1<H && mask[(r+1)*W + c]){
        float nc = cost[(r+1)*W + c];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + c]; }
    }
    if (c>0 && mask[r*W + (c-1)]){
        float nc = cost[r*W + (c-1)];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[r*W + (c-1)]; }
    }
    if (c+1<W && mask[r*W + (c+1)]){
        float nc = cost[r*W + (c+1)];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[r*W + (c+1)]; }
    }
    if (use8){
        if (r>0 && c>0 && mask[(r-1)*W + (c-1)]){
            float nc = cost[(r-1)*W + (c-1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + (c-1)]; }
        }
        if (r>0 && c+1<W && mask[(r-1)*W + (c+1)]){
            float nc = cost[(r-1)*W + (c+1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + (c+1)]; }
        }
        if (r+1<H && c>0 && mask[(r+1)*W + (c-1)]){
            float nc = cost[(r+1)*W + (c-1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + (c-1)]; }
        }
        if (r+1<H && c+1<W && mask[(r+1)*W + (c+1)]){
            float nc = cost[(r+1)*W + (c+1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + (c+1)]; }
        }
    }
    depth[idx] = best_depth; cost[idx] = best_cost;
}

extern "C" __global__ void random_search_depth(
    float* depth, const unsigned char* mask,
    const int H, const int W, const float step)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    const int idx = r*W + c;
    float d = depth[idx];
    if (!(d>0.0f) || isnan(d) || isinf(d)) return;
    float sign = ((r ^ c) & 1) ? 1.0f : -1.0f;
    float dn = d + sign * step;
    if (dn > 0.0f) depth[idx] = dn;
}

extern "C" __global__ void compute_cost_grad(
    const float* img, float* cost, const int H, const int W)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    float gx = 0.0f, gy = 0.0f;
    #define I(rr,cc) img[(rr)*W + (cc)]
    if (r>0 && r+1<H && c>0 && c+1<W){
        gx = -I(r-1,c-1) - 2.0f*I(r,c-1) - I(r+1,c-1)
             + I(r-1,c+1) + 2.0f*I(r,c+1) + I(r+1,c+1);
        gy = -I(r-1,c-1) - 2.0f*I(r-1,c) - I(r-1,c+1)
             + I(r+1,c-1) + 2.0f*I(r+1,c) + I(r+1,c+1);
    }
    float g = fabsf(gx) + fabsf(gy);
    cost[r*W + c] = 1.0f / (1.0f + g);
}
extern "C" __global__ void propagate_checker4(
    float* depth, float* cost,
    const unsigned char* mask, const int H, const int W, const int color, const int use8)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    if (((r+c)&1) != color) return;

    const int idx = r*W + c;
    float best_cost = cost[idx];
    float best_depth = depth[idx];

    // up
    if (r>0 && mask[(r-1)*W + c]){
        float nc = cost[(r-1)*W + c];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + c]; }
    }
    // down
    if (r+1<H && mask[(r+1)*W + c]){
        float nc = cost[(r+1)*W + c];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + c]; }
    }
    // left
    if (c>0 && mask[r*W + (c-1)]){
        float nc = cost[r*W + (c-1)];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[r*W + (c-1)]; }
    }
    // right
    if (c+1<W && mask[r*W + (c+1)]){
        float nc = cost[r*W + (c+1)];
        if (nc < best_cost){ best_cost = nc; best_depth = depth[r*W + (c+1)]; }
    }

    if (use8){
        // diagonals
        if (r>0 && c>0 && mask[(r-1)*W + (c-1)]){
            float nc = cost[(r-1)*W + (c-1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + (c-1)]; }
        }
        if (r>0 && c+1<W && mask[(r-1)*W + (c+1)]){
            float nc = cost[(r-1)*W + (c+1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r-1)*W + (c+1)]; }
        }
        if (r+1<H && c>0 && mask[(r+1)*W + (c-1)]){
            float nc = cost[(r+1)*W + (c-1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + (c-1)]; }
        }
        if (r+1<H && c+1<W && mask[(r+1)*W + (c+1)]){
            float nc = cost[(r+1)*W + (c+1)];
            if (nc < best_cost){ best_cost = nc; best_depth = depth[(r+1)*W + (c+1)]; }
        }
    }

    depth[idx] = best_depth;
    cost[idx]  = best_cost;
}

extern "C" __global__ void random_search_depth(
    float* depth, const unsigned char* mask,
    const int H, const int W, const float step)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    const int idx = r*W + c;
    float d = depth[idx];
    if (!(d>0.0f) || isnan(d) || isinf(d)) return;
    // 乱数の代わりに座標由来の擬似摂動（決定的）
    float sign = ((r ^ c) & 1) ? 1.0f : -1.0f;
    float delta = sign * step;
    float dn = d + delta;
    if (dn > 0.0f) depth[idx] = dn;
}

// ref座標の深度を3D化→world→各srcへ投影→|I_ref - I_src|の平均をcostに格納
__device__ __forceinline__ float bilinear(const float* img, int H, int W, float v, float u){
    if (u < 0.0f || v < 0.0f || u > (float)(W-1) || v > (float)(H-1)) return 0.0f;
    int u0 = (int)floorf(u), v0 = (int)floorf(v);
    int u1 = min(u0+1, W-1), v1 = min(v0+1, H-1);
    float du = u - (float)u0, dv = v - (float)v0;
    float I00 = img[v0*W+u0];
    float I10 = img[v0*W+u1];
    float I01 = img[v1*W+u0];
    float I11 = img[v1*W+u1];
    return (1-du)*(1-dv)*I00 + du*(1-dv)*I10 + (1-du)*dv*I01 + du*dv*I11;
}

extern "C" __global__ void compute_cost_project(
    const float* ref_gray, const float* src_imgs, // [N,H,W]
    const float* K_ref, const float* R_ref, const float* T_ref,
    const float* src_K, const float* src_R, const float* src_T,
    const float* depth, float* cost, const int H, const int W, const int N,
    const int topk, const int use_median, const int patch_size,
    const float sigma_color, const float zncc_eps)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    const int idx = r*W + c;
    float d = depth[idx];
    if (!(d>0.0f)) { cost[idx] = 1.0f; return; }
    // ref: pixel -> camera
    float fx = K_ref[0], fy = K_ref[4], cx = K_ref[2], cy = K_ref[5];
    float x = ( (float)c - cx ) * d / fx;
    float y = ( (float)r - cy ) * d / fy;
    float z = d;
    // camera -> world (R is 3x3 row-major)
    // R_ref^T * (p - T)
    float px = x - T_ref[0];
    float py = y - T_ref[1];
    float pz = z - T_ref[2];
    float Rrt00 = R_ref[0], Rrt01 = R_ref[3], Rrt02 = R_ref[6];
    float Rrt10 = R_ref[1], Rrt11 = R_ref[4], Rrt12 = R_ref[7];
    float Rrt20 = R_ref[2], Rrt21 = R_ref[5], Rrt22 = R_ref[8];
    float wx = Rrt00*px + Rrt01*py + Rrt02*pz;
    float wy = Rrt10*px + Rrt11*py + Rrt12*pz;
    float wz = Rrt20*px + Rrt21*py + Rrt22*pz;

    // collect up to 32 diffs (assume N<=32)
    float diffs[32]; int dcnt = 0;
    for(int n=0;n<N;++n){
        const float* Kr = src_K + 9*n;
        const float* Rr = src_R + 9*n;
        const float* Tr = src_T + 3*n;
        float sx = Rr[0]*wx + Rr[1]*wy + Rr[2]*wz + Tr[0];
        float sy = Rr[3]*wx + Rr[4]*wy + Rr[5]*wz + Tr[1];
        float sz = Rr[6]*wx + Rr[7]*wy + Rr[8]*wz + Tr[2];
        if (fabsf(sz) < 1e-6f) continue;
        float u = Kr[0]*sx/sz + Kr[2];
        float v = Kr[4]*sy/sz + Kr[5];
        const float* img = src_imgs + n*(H*W);
        // weighted ZNCC on patch
        int half = patch_size/2;
        float sumw = 0.0f;
        float sx=0.0f, sy=0.0f, sxx=0.0f, syy=0.0f, sxy=0.0f;
        for(int pr=-half; pr<=half; ++pr){
            for(int pc=-half; pc<=half; ++pc){
                float wr = (float)(r+pr);
                float wc = (float)(c+pc);
                float I0 = 0.0f;
                if (wr>=0.0f && wr< (float)H && wc>=0.0f && wc<(float)W){
                    I0 = bilinear(ref_gray, H, W, wr, wc);
                }
                float u_ref = wc; float v_ref = wr;
                float p0 = Rrt00*( (u_ref-cx)*d/fx ) + Rrt01*( (v_ref-cy)*d/fy ) + Rrt02*d + (-Rrt00*T_ref[0]-Rrt01*T_ref[1]-Rrt02*T_ref[2]);
                float p1 = Rrt10*( (u_ref-cx)*d/fx ) + Rrt11*( (v_ref-cy)*d/fy ) + Rrt12*d + (-Rrt10*T_ref[0]-Rrt11*T_ref[1]-Rrt12*T_ref[2]);
                float p2 = Rrt20*( (u_ref-cx)*d/fx ) + Rrt21*( (v_ref-cy)*d/fy ) + Rrt22*d + (-Rrt20*T_ref[0]-Rrt21*T_ref[1]-Rrt22*T_ref[2]);
                // world->src
                float sx0 = Rr[0]*p0 + Rr[1]*p1 + Rr[2]*p2 + Tr[0];
                float sy0 = Rr[3]*p0 + Rr[4]*p1 + Rr[5]*p2 + Tr[1];
                float sz0 = Rr[6]*p0 + Rr[7]*p1 + Rr[8]*p2 + Tr[2];
                if (fabsf(sz0) < 1e-6f) continue;
                float uu = Kr[0]*sx0/sz0 + Kr[2];
                float vv = Kr[4]*sy0/sz0 + Kr[5];
                float I1 = bilinear(img, H, W, vv, uu);
                // gaussian weight by color difference (approx): here use spatially uniform for simplicity
                float w = 1.0f; // could incorporate color/adaptive weight if needed
                sumw += w;
                sx += w*I0; sy += w*I1;
                sxx += w*I0*I0; syy += w*I1*I1; sxy += w*I0*I1;
            }
        }
        if (sumw <= 1e-6f) continue;
        float mx = sx/sumw, my = sy/sumw;
        float vx = max(0.0f, sxx/sumw - mx*mx);
        float vy = max(0.0f, syy/sumw - my*my);
        float denom = sqrtf(vx*vy);
        float zncc = (denom > zncc_eps) ? ((sxy/sumw - mx*my) / denom) : 0.0f;
        float costv = 0.5f * (1.0f - zncc);
        if (dcnt < 32){ diffs[dcnt++] = costv; }
    }
    if (dcnt==0){ cost[idx] = 1.0f; return; }
    // sort small array (selection sort)
    for(int i=0;i<dcnt;i++){
        int mi=i; float mv=diffs[i];
        for(int j=i+1;j<dcnt;j++){ if (diffs[j] < mv){ mi=j; mv=diffs[j]; } }
        float tmp=diffs[i]; diffs[i]=diffs[mi]; diffs[mi]=tmp;
    }
    int k = topk < dcnt ? topk : dcnt;
    if (use_median){
        int mid = k/2; cost[idx] = diffs[mid];
    } else {
        float acc=0.0f; for(int i=0;i<k;i++) acc += diffs[i];
        cost[idx] = acc / (float)k;
    }
}

// ACMH-lite: H仮説のベスト/セカンドを近傍も含めて選択し、slot0/slot1に反映
extern "C" __global__ void propagate_acmhH(
    float* depthH, float* costH, const unsigned char* mask,
    const int Hs, const int H, const int W, const int color, const int use8)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    if (((r+c)&1) != color) return;
    const int HW = H*W;
    const int idx = r*W + c;

    float best_cost = 1e9f, second_cost = 1e9f;
    float best_depth = NAN, second_depth = NAN;

    // self slots
    for(int s=0;s<Hs;++s){
        float cs = costH[s*HW + idx];
        float ds = depthH[s*HW + idx];
        if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
        else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
    }
    // neighbors
    // up/down/left/right
    int rr, cc, nidx;
    // up
    rr=r-1; cc=c; if (rr>=0 && mask[rr*W+cc]){
        nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx];
            if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
            else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
        }
    }
    // down
    rr=r+1; cc=c; if (rr<H && mask[rr*W+cc]){
        nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx];
            if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
            else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
        }
    }
    // left
    rr=r; cc=c-1; if (cc>=0 && mask[rr*W+cc]){
        nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx];
            if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
            else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
        }
    }
    // right
    rr=r; cc=c+1; if (cc<W && mask[rr*W+cc]){
        nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx];
            if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
            else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
        }
    }
    if (use8){
        // diag
        rr=r-1; cc=c-1;
        if (rr>=0 && cc>=0 && mask[rr*W+cc]){
            nidx=rr*W+cc;
            for(int s=0;s<Hs;++s){
                float cs=costH[s*HW+nidx];
                float ds=depthH[s*HW+nidx];
                if (cs < best_cost){
                    second_cost=best_cost; second_depth=best_depth;
                    best_cost=cs; best_depth=ds;
                } else if (cs < second_cost){
                    second_cost=cs; second_depth=ds;
                }
            }
        }
        rr=r-1; cc=c+1;
        if (rr>=0 && cc<W && mask[rr*W+cc]){
            nidx=rr*W+cc;
            for(int s=0;s<Hs;++s){
                float cs=costH[s*HW+nidx];
                float ds=depthH[s*HW+nidx];
                if (cs < best_cost){
                    second_cost=best_cost; second_depth=best_depth;
                    best_cost=cs; best_depth=ds;
                } else if (cs < second_cost){
                    second_cost=cs; second_depth=ds;
                }
            }
        }
        rr=r+1; cc=c-1;
        if (rr<H && cc>=0 && mask[rr*W+cc]){
            nidx=rr*W+cc;
            for(int s=0;s<Hs;++s){
                float cs=costH[s*HW+nidx];
                float ds=depthH[s*HW+nidx];
                if (cs < best_cost){
                    second_cost=best_cost; second_depth=best_depth;
                    best_cost=cs; best_depth=ds;
                } else if (cs < second_cost){
                    second_cost=cs; second_depth=ds;
                }
            }
        }
        rr=r+1; cc=c+1;
        if (rr<H && cc<W && mask[rr*W+cc]){
            nidx=rr*W+cc;
            for(int s=0;s<Hs;++s){
                float cs=costH[s*HW+nidx];
                float ds=depthH[s*HW+nidx];
                if (cs < best_cost){
                    second_cost=best_cost; second_depth=best_depth;
                    best_cost=cs; best_depth=ds;
                } else if (cs < second_cost){
                    second_cost=cs; second_depth=ds;
                }
            }
        }
    }

    // write back to slot0/slot1 if available
    depthH[0*HW + idx] = best_depth; costH[0*HW + idx] = best_cost;
    if (Hs>1){ depthH[1*HW + idx] = second_depth; costH[1*HW + idx] = second_cost; }
}
extern "C" __global__ void compute_cost_grad(
    const float* img, float* cost, const int H, const int W)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    float gx = 0.0f, gy = 0.0f;
    #define I(rr,cc) img[(rr)*W + (cc)]
    if (r>0 && r+1<H && c>0 && c+1<W){
        gx = -I(r-1,c-1) - 2.0f*I(r,c-1) - I(r+1,c-1)
             + I(r-1,c+1) + 2.0f*I(r,c+1) + I(r+1,c+1);
        gy = -I(r-1,c-1) - 2.0f*I(r-1,c) - I(r-1,c+1)
             + I(r+1,c-1) + 2.0f*I(r+1,c) + I(r+1,c+1);
    }
    float g = fabsf(gx) + fabsf(gy);
    cost[r*W + c] = 1.0f / (1.0f + g);
}
"""
        self.mod = cp.RawModule(
            code=src,
            options=("-std=c++11",),
            name_expressions=(
                "propagate_checker4",
                "random_search_depth",
                "compute_cost_grad",
                "compute_cost_project",
                "propagate_acmhH",
            ),
        )
        DepthOptimization._raw_module = self.mod
        self._bind_functions()

    def _bind_functions(self):
        self.k_prop4 = self.mod.get_function("propagate_checker4")
        self.k_rand = self.mod.get_function("random_search_depth")
        self.k_cost = self.mod.get_function("compute_cost_grad")
        self.k_cost_proj = self.mod.get_function("compute_cost_project")
        self.k_acmh = self.mod.get_function("propagate_acmhH")
        # new: eval-inlined variants
        try:
            self.k_prop4_eval = self.mod.get_function("propagate_checker4_eval")
            self.k_rand_eval = self.mod.get_function("random_search_depth_eval")
        except Exception:
            self.k_prop4_eval = None
            self.k_rand_eval = None

    def _warmup(self):
        H, W = 16, 16
        depth = cp.ones((H, W), dtype=cp.float32)
        cost = cp.ones((H, W), dtype=cp.float32)
        mask = cp.ones((H, W), dtype=cp.uint8)
        img = cp.zeros((H, W), dtype=cp.float32)
        threads = (16, 16)
        blocks = (1, 1)
        self.k_cost(blocks, threads, (img, cost, np.int32(H), np.int32(W)))
        self.k_prop4(
            blocks,
            threads,
            (depth, cost, mask, np.int32(H), np.int32(W), np.int32(0), np.int32(0)),
        )
        self.k_rand(
            blocks, threads, (depth, mask, np.int32(H), np.int32(W), np.float32(0.01))
        )
        cp.cuda.Stream.null.synchronize()

    def refine_depth_with_patchmatch(self, **kwargs):
        # CuPy簡易版: 初期コスト（initial_depth_error）に基づくcheckerboard拡散（評価は隣接最小コスト伝播）
        initial_depth = kwargs.get("initial_depth")
        initial_depth_error = kwargs.get("initial_depth_error")
        ref_idx = kwargs.get("ref_idx", -1)
        ref_image = kwargs.get("ref_image")
        neighbor_views = kwargs.get("neighbor_views_data", [])
        ref_pose = kwargs.get("ref_pose")

        if initial_depth is None or initial_depth_error is None:
            logging.warning(
                "CuPy backend received no initial_depth/initial_depth_error. Falling back."
            )
            from .depth_optimization_gpu import DepthOptimization as NumbaDepthOpt

            return NumbaDepthOpt(self.config).refine_depth_with_patchmatch(**kwargs)

        h, w = initial_depth.shape
        depth = cp.asarray(initial_depth.astype(np.float32))
        if initial_depth_error is not None:
            cost = cp.asarray(initial_depth_error.astype(np.float32))
        else:
            cost = cp.ones((h, w), dtype=cp.float32)
        # 法線マップ（Z軸初期化）
        normal = self._get_buf("normal", (h, w, 3), cp.float32)
        normal[...] = 0.0
        normal[:, :, 2] = 1.0
        mask_np = np.isfinite(initial_depth).astype(np.uint8)
        mask = cp.asarray(mask_np)

        threads = (16, 16)
        blocks = (
            (w + threads[0] - 1) // threads[0],
            (h + threads[1] - 1) // threads[1],
        )

        iters = int(getattr(self.config, "PATCHMATCH_ITERATIONS", 5))
        decay = float(getattr(self.config, "PATCHMATCH_DECAY_RATE", 0.9))
        step0 = float(
            max(
                1e-3,
                getattr(self.config, "PATCHMATCH_VANILLA_INITIAL_SEARCH_RANGE", 50.0)
                * 0.001,
            )
        )
        # 参照画像と近傍データを事前にGPUへ持ち込み
        ref_gray = None
        try:
            if ref_image is not None:
                import cv2

                if ref_image.ndim == 3:
                    rg = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                else:
                    rg = ref_image.astype(np.float32)
                if rg.shape != (h, w):
                    rg = cv2.resize(rg, (w, h), interpolation=cv2.INTER_LINEAR)
                ref_gray = cp.asarray(rg)
                self.k_cost(blocks, threads, (ref_gray, cost, np.int32(h), np.int32(w)))
        except Exception as e:
            logging.debug(f"CuPy cost kernel skipped: {e}")

        src_imgs_cp = None
        K_ref = None
        R_ref = None
        T_ref = None
        srcK_cp = None
        srcR_cp = None
        srcT_cp = None
        try:
            if neighbor_views and ref_gray is not None and ref_pose is not None:
                N = len(neighbor_views)
                import cv2

                src_imgs = np.empty((N, h, w), dtype=np.float32)
                src_K = np.empty((N, 9), dtype=np.float32)
                src_R = np.empty((N, 9), dtype=np.float32)
                src_T = np.empty((N, 3), dtype=np.float32)
                indices = tuple(int(v.get("image_idx", -1)) for v in neighbor_views)
                for i, v in enumerate(neighbor_views):
                    img = v["image"]
                    g = (
                        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        if img.ndim == 3
                        else img.astype(np.float32)
                    )
                    if g.shape != (h, w):
                        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
                    src_imgs[i] = g
                    src_K[i] = v["K"].astype(np.float32).reshape(-1)
                    src_R[i] = v["R"].astype(np.float32).reshape(-1)
                    src_T[i] = v["T"].astype(np.float32).reshape(-1)
                # 参照行列はフレームごとに更新されるので都度アップロード
                K_ref = cp.asarray(ref_pose["K"].astype(np.float32).reshape(-1))
                R_ref = cp.asarray(ref_pose["R"].astype(np.float32).reshape(-1))
                T_ref = cp.asarray(ref_pose["T"].astype(np.float32).reshape(-1))
                # 近傍K/R/Tはインデックスと解像度が同じならキャッシュを再利用
                if (
                    self._cam_cache["indices"] == indices
                    and self._cam_cache["h"] == h
                    and self._cam_cache["w"] == w
                    and self._cam_cache["srcK_cp"] is not None
                ):
                    srcK_cp = self._cam_cache["srcK_cp"]
                    srcR_cp = self._cam_cache["srcR_cp"]
                    srcT_cp = self._cam_cache["srcT_cp"]
                else:
                    srcK_cp = cp.asarray(src_K.reshape(-1))
                    srcR_cp = cp.asarray(src_R.reshape(-1))
                    srcT_cp = cp.asarray(src_T.reshape(-1))
                    self._cam_cache.update(
                        {
                            "indices": indices,
                            "h": h,
                            "w": w,
                            "srcK_cp": srcK_cp,
                            "srcR_cp": srcR_cp,
                            "srcT_cp": srcT_cp,
                        }
                    )
                # 画像スタックは毎回内容が変わるので都度コピー（バッファは再利用）
                src_imgs_cp = self._get_buf("src_imgs_cp", (N, h, w), cp.float32)
                src_imgs_cp.set(src_imgs)
        except Exception as e:
            logging.debug(f"CuPy neighbor prefetch skipped: {e}")
        use8 = int(getattr(self.config, "PROPAGATION_NEIGHBOR_DIRECTIONS", 4) == 8)
        topk = int(getattr(self.config, "TOP_K_COSTS", 5))
        use_median = int(getattr(self.config, "USE_MEDIAN_TOP_K", 1))
        acmh_enable = int(getattr(self.config, "ACMH_ENABLE", 0))
        Hs = int(getattr(self.config, "ACMH_NUM_HYPOTHESES", 2)) if acmh_enable else 1

        # メトリクス/CSV ログ（CuPy経路）
        csv_files = kwargs.get("csv_files")
        iter_times = kwargs.get("iter_times")
        # save_dir is optionally used when DEBUG_SAVE_DEPTH_MAPS; don't bind if unused
        from .utils import append_to_csv

        # 評価/可視化ユーティリティ（必要時のみ使用）
        gt_depth = kwargs.get("gt_depth")
        ref_idx = kwargs.get("ref_idx", -1)
        try:
            import config as _app_config

            from .utils import (
                compute_depth_metrics,
                save_depth_map_as_image,
                save_error_map_as_image,
            )
        except Exception:
            compute_depth_metrics = None
            save_error_map_as_image = None
            save_depth_map_as_image = None
            _app_config = None

        if Hs <= 1:
            import time

            start_iter = time.time()
            for i in range(iters):
                logging.info(f"[CuPy] Iter {i+1}/{iters} - start")
                for color in (0, 1):
                    if (
                        self.k_prop4_eval is not None
                        and src_imgs_cp is not None
                        and ref_gray is not None
                    ):
                        self.k_prop4_eval(
                            blocks,
                            threads,
                            (
                                depth,
                                cost,
                                normal,
                                mask,
                                np.int32(h),
                                np.int32(w),
                                np.int32(color),
                                np.int32(use8),
                                ref_gray,
                                src_imgs_cp,
                                K_ref,
                                R_ref,
                                T_ref,
                                srcK_cp,
                                srcR_cp,
                                srcT_cp,
                                np.int32(src_imgs_cp.shape[0]),
                                np.int32(topk),
                                np.int32(use_median),
                                np.int32(
                                    getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)
                                ),
                                np.float32(
                                    getattr(
                                        self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0
                                    )
                                ),
                                np.float32(getattr(self.config, "ZNCC_EPSILON", 1e-6)),
                                np.float32(
                                    getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                                ),
                                np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                            ),
                        )
                    else:
                        self.k_prop4(
                            blocks,
                            threads,
                            (
                                depth,
                                cost,
                                mask,
                                np.int32(h),
                                np.int32(w),
                                np.int32(color),
                                np.int32(use8),
                            ),
                        )
                # ランダムサーチ相当（簡易摂動）
                step0 *= decay
                # depth_range (per-pixel) を用意
                try:
                    depth_range_np = (initial_depth_error * (decay**i)).astype(
                        np.float32
                    )
                except Exception:
                    depth_range_np = np.full((h, w), float(step0), dtype=np.float32)
                depth_range_cp = self._get_buf("depth_range_cp", (h, w), cp.float32)
                depth_range_cp.set(depth_range_np)
                if (
                    self.k_rand_eval is not None
                    and src_imgs_cp is not None
                    and ref_gray is not None
                ):
                    self.k_rand_eval(
                        blocks,
                        threads,
                        (
                            depth,
                            cost,
                            normal,
                            mask,
                            np.int32(h),
                            np.int32(w),
                            depth_range_cp,
                            ref_gray,
                            src_imgs_cp,
                            K_ref,
                            R_ref,
                            T_ref,
                            srcK_cp,
                            srcR_cp,
                            srcT_cp,
                            np.int32(src_imgs_cp.shape[0]),
                            np.int32(topk),
                            np.int32(use_median),
                            np.int32(getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)),
                            np.float32(
                                getattr(
                                    self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0
                                )
                            ),
                            np.float32(getattr(self.config, "ZNCC_EPSILON", 1e-6)),
                            np.int32((i + 1) * 1664525),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_VANILLA_MIN_DEPTH", 1.0
                                )
                            ),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_VANILLA_MAX_DEPTH", 100.0
                                )
                            ),
                            np.float32(
                                getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                            ),
                            np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_NORMAL_SEARCH_ANGLE", 20.0
                                )
                            ),
                            np.float32(
                                getattr(self.config, "PATCHMATCH_DECAY_RATE", 0.9) ** i
                            ),
                        ),
                    )
                else:
                    self.k_rand(
                        blocks,
                        threads,
                        (depth, mask, np.int32(h), np.int32(w), np.float32(step0)),
                    )
                # 投影コスト更新（近傍がある場合）
                try:
                    if src_imgs_cp is not None and ref_gray is not None:
                        self.k_cost_proj(
                            blocks,
                            threads,
                            (
                                ref_gray,
                                src_imgs_cp,
                                K_ref,
                                R_ref,
                                T_ref,
                                srcK_cp,
                                srcR_cp,
                                srcT_cp,
                                depth,
                                cost,
                                np.int32(h),
                                np.int32(w),
                                np.int32(src_imgs_cp.shape[0]),
                                np.int32(topk),
                                np.int32(use_median),
                                np.int32(
                                    getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)
                                ),
                                np.float32(
                                    getattr(
                                        self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0
                                    )
                                ),
                                np.float32(getattr(self.config, "ZNCC_EPSILON", 1e-6)),
                            ),
                        )
                except Exception as e:
                    logging.debug(f"CuPy projection cost update skipped: {e}")
                try:
                    cmean_iter = float(cp.nanmean(cost).get())
                    logging.info(
                        f"[CuPy] Iter {i+1}/{iters} - mean cost: {cmean_iter:.6f}"
                    )
                except Exception:
                    pass
                # 反復時間計測とCSV出力
                if iter_times is not None:
                    iter_times.append(time.time() - start_iter)
                    start_iter = time.time()
                if csv_files is not None:
                    try:
                        # ここではコストの平均を疑似メトリクスとして出す（本来はMAE等を計算）
                        cmean = float(cp.nanmean(cost).get())
                        if "mae" in csv_files:
                            append_to_csv(
                                csv_files["mae"],
                                [sum(iter_times) if iter_times else 0.0, cmean],
                            )
                    except Exception:
                        pass
                # 反復ごとの評価と保存
                if compute_depth_metrics is not None and gt_depth is not None:
                    try:
                        depth_np = cp.asnumpy(depth)
                        metrics_it = compute_depth_metrics(depth_np, gt_depth)
                        logging.info(
                            f"[CuPy] Iter {i+1}/{iters} - MAE: {metrics_it['mae']:.4f}, AbsRel: {metrics_it['abs_rel']:.4f}"
                        )
                        if (
                            getattr(self.config, "DEBUG_SAVE_DEPTH_MAPS", False)
                            and _app_config is not None
                        ):
                            import os as _os

                            save_each_depth_dir = _os.path.join(
                                _app_config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
                            )
                            _os.makedirs(save_each_depth_dir, exist_ok=True)
                            if save_depth_map_as_image is not None:
                                save_depth_map_as_image(
                                    depth_np.copy(),
                                    _os.path.join(
                                        save_each_depth_dir, f"depth_iter_{i+1:02d}.png"
                                    ),
                                )
                            if save_error_map_as_image is not None:
                                save_error_map_as_image(
                                    depth_np,
                                    gt_depth,
                                    _os.path.join(
                                        save_each_depth_dir,
                                        f"error_map_iter_{i+1:02d}.png",
                                    ),
                                )
                    except Exception:
                        pass
        else:
            # Hスロット確保（slot0=現行、他は複製）
            depthH = self._get_buf("depthH", (Hs, h, w), cp.float32)
            costH = self._get_buf("costH", (Hs, h, w), cp.float32)
            depthH[0] = depth
            costH[0] = cost
            for s in range(1, Hs):
                depthH[s] = depth
                costH[s] = cost
            depthH_ = depthH.reshape(Hs, -1)
            costH_ = costH.reshape(Hs, -1)
            import time

            start_iter = time.time()
            for i in range(iters):
                logging.info(f"[CuPy][H={Hs}] Iter {i+1}/{iters} - start")
                # 各色でACMH-lite選抜（slot0/slot1更新）
                for color in (0, 1):
                    self.k_acmh(
                        blocks,
                        threads,
                        (
                            depthH_.data.ptr,
                            costH_.data.ptr,
                            mask,
                            np.int32(Hs),
                            np.int32(h),
                            np.int32(w),
                            np.int32(color),
                            np.int32(use8),
                        ),
                    )
                # slot0をcurrentに反映
                depth = depthH[0]
                cost = costH[0]
                # slot0に対して評価付きcheckerboard伝播を適用（利用可能時）
                for color in (0, 1):
                    if (
                        self.k_prop4_eval is not None
                        and src_imgs_cp is not None
                        and ref_gray is not None
                    ):
                        self.k_prop4_eval(
                            blocks,
                            threads,
                            (
                                depth,
                                cost,
                                normal,
                                mask,
                                np.int32(h),
                                np.int32(w),
                                np.int32(color),
                                np.int32(use8),
                                ref_gray,
                                src_imgs_cp,
                                K_ref,
                                R_ref,
                                T_ref,
                                srcK_cp,
                                srcR_cp,
                                srcT_cp,
                                np.int32(src_imgs_cp.shape[0]),
                                np.int32(topk),
                                np.int32(use_median),
                                np.int32(
                                    getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)
                                ),
                                np.float32(
                                    getattr(
                                        self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0
                                    )
                                ),
                                np.float32(getattr(self.config, "ZNCC_EPSILON", 1e-6)),
                                np.float32(
                                    getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                                ),
                                np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                            ),
                        )
                    else:
                        self.k_prop4(
                            blocks,
                            threads,
                            (
                                depth,
                                cost,
                                mask,
                                np.int32(h),
                                np.int32(w),
                                np.int32(color),
                                np.int32(use8),
                            ),
                        )
                # 摂動
                step0 *= decay
                # depth_range (per-pixel)
                try:
                    depth_range_np = (initial_depth_error * (decay**i)).astype(
                        np.float32
                    )
                except Exception:
                    depth_range_np = np.full((h, w), float(step0), dtype=np.float32)
                depth_range_cp = self._get_buf("depth_range_cp", (h, w), cp.float32)
                depth_range_cp.set(depth_range_np)
                if (
                    self.k_rand_eval is not None
                    and src_imgs_cp is not None
                    and ref_gray is not None
                ):
                    self.k_rand_eval(
                        blocks,
                        threads,
                        (
                            depth,
                            cost,
                            normal,
                            mask,
                            np.int32(h),
                            np.int32(w),
                            depth_range_cp,
                            ref_gray,
                            src_imgs_cp,
                            K_ref,
                            R_ref,
                            T_ref,
                            srcK_cp,
                            srcR_cp,
                            srcT_cp,
                            np.int32(src_imgs_cp.shape[0]),
                            np.int32(topk),
                            np.int32(use_median),
                            np.int32(getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)),
                            np.float32(
                                getattr(
                                    self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR", 10.0
                                )
                            ),
                            np.float32(getattr(self.config, "ZNCC_EPSILON", 1e-6)),
                            np.int32((i + 1) * 1664525),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_VANILLA_MIN_DEPTH", 1.0
                                )
                            ),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_VANILLA_MAX_DEPTH", 100.0
                                )
                            ),
                            np.float32(
                                getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                            ),
                            np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                            np.float32(
                                getattr(
                                    self.config, "PATCHMATCH_NORMAL_SEARCH_ANGLE", 20.0
                                )
                            ),
                            np.float32(
                                getattr(self.config, "PATCHMATCH_DECAY_RATE", 0.9) ** i
                            ),
                        ),
                    )
                else:
                    self.k_rand(
                        blocks,
                        threads,
                        (depth, mask, np.int32(h), np.int32(w), np.float32(step0)),
                    )
                # 各スロットの投影コストを個別に更新
                try:
                    if src_imgs_cp is not None and ref_gray is not None:
                        for s in range(Hs):
                            self.k_cost_proj(
                                blocks,
                                threads,
                                (
                                    ref_gray,
                                    src_imgs_cp,
                                    K_ref,
                                    R_ref,
                                    T_ref,
                                    srcK_cp,
                                    srcR_cp,
                                    srcT_cp,
                                    depthH[s],
                                    costH[s],
                                    np.int32(h),
                                    np.int32(w),
                                    np.int32(src_imgs_cp.shape[0]),
                                    np.int32(topk),
                                    np.int32(use_median),
                                    np.int32(
                                        getattr(self.config, "PATCHMATCH_PATCH_SIZE", 7)
                                    ),
                                    np.float32(
                                        getattr(
                                            self.config,
                                            "ADAPTIVE_WEIGHT_SIGMA_COLOR",
                                            10.0,
                                        )
                                    ),
                                    np.float32(
                                        getattr(self.config, "ZNCC_EPSILON", 1e-6)
                                    ),
                                ),
                            )
                except Exception as e:
                    logging.debug(f"CuPy projection cost update skipped: {e}")
                try:
                    cmean_iter = float(cp.nanmean(costH[0]).get())
                    logging.info(
                        f"[CuPy][H={Hs}] Iter {i+1}/{iters} - mean cost: {cmean_iter:.6f}"
                    )
                except Exception:
                    pass
                # 反復時間計測とCSV出力
                if iter_times is not None:
                    iter_times.append(time.time() - start_iter)
                    start_iter = time.time()
                if csv_files is not None:
                    try:
                        cmean = float(cp.nanmean(costH[0]).get())
                        if "mae" in csv_files:
                            append_to_csv(
                                csv_files["mae"],
                                [sum(iter_times) if iter_times else 0.0, cmean],
                            )
                    except Exception:
                        pass
                # 反復ごとの評価と保存（slot0）
                if compute_depth_metrics is not None and gt_depth is not None:
                    try:
                        depth_np = cp.asnumpy(depthH[0])
                        metrics_it = compute_depth_metrics(depth_np, gt_depth)
                        logging.info(
                            f"[CuPy][H={Hs}] Iter {i+1}/{iters} - MAE: {metrics_it['mae']:.4f}, AbsRel: {metrics_it['abs_rel']:.4f}"
                        )
                        if (
                            getattr(self.config, "DEBUG_SAVE_DEPTH_MAPS", False)
                            and _app_config is not None
                        ):
                            import os as _os

                            save_each_depth_dir = _os.path.join(
                                _app_config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
                            )
                            _os.makedirs(save_each_depth_dir, exist_ok=True)
                            if save_depth_map_as_image is not None:
                                save_depth_map_as_image(
                                    depth_np.copy(),
                                    _os.path.join(
                                        save_each_depth_dir, f"depth_iter_{i+1:02d}.png"
                                    ),
                                )
                            if save_error_map_as_image is not None:
                                save_error_map_as_image(
                                    depth_np,
                                    gt_depth,
                                    _os.path.join(
                                        save_each_depth_dir,
                                        f"error_map_iter_{i+1:02d}.png",
                                    ),
                                )
                    except Exception:
                        pass
        cp.cuda.Stream.null.synchronize()

        optimized = cp.asnumpy(depth)
        return optimized

    def filter_depth_map_by_photometric_consistency(
        self, depth_map, ref_image, ref_pose, neighbor_views_data
    ):
        """
        光度一貫性に基づくフィルタ（GPU実装と同等のロジック）
        近傍ビューに投影したパッチの色差が小さいビュー数がしきい値未満なら該当画素を無効化。
        """
        import logging as _logging

        import numpy as _np

        _logging.info(
            "Filtering optimized depth map based on cost and photometric consistency..."
        )
        if depth_map is None:
            return depth_map

        h, w = depth_map.shape
        filtered = depth_map.copy()

        # 参照姿勢
        K = ref_pose["K"].astype(_np.float32)
        R_ref = ref_pose["R"].astype(_np.float32)
        T_ref = ref_pose["T"].astype(_np.float32)

        # 画像（RGB/Gray対応）
        ref_img = ref_image.astype(_np.float32)
        if ref_img.ndim == 2:
            # Gray -> 3ch化
            ref_img = _np.stack([ref_img, ref_img, ref_img], axis=-1)

        # 近傍ビュー準備
        if not neighbor_views_data:
            return filtered
        neighbor_images = []
        neighbor_R = []
        neighbor_T = []
        for view in neighbor_views_data:
            img = view["image"].astype(_np.float32)
            if img.ndim == 2:
                img = _np.stack([img, img, img], axis=-1)
            neighbor_images.append(img)
            neighbor_R.append(view["R"].astype(_np.float32))
            neighbor_T.append(view["T"].astype(_np.float32))
        neighbor_images = _np.stack(neighbor_images, axis=0)
        neighbor_R = _np.stack(neighbor_R, axis=0)
        neighbor_T = _np.stack(neighbor_T, axis=0)

        thresh = float(
            getattr(self.config, "FILTERING_COLOR_DIFFERENCE_THRESHOLD", 20.0)
        )
        min_consistent = int(getattr(self.config, "FILTERING_MIN_CONSISTENT_VIEWS", 3))

        def _bilinear_color(img, v, u):
            h_, w_, _ = img.shape
            if u < 0 or v < 0 or u > (w_ - 1) or v > (h_ - 1):
                return _np.zeros((3,), dtype=_np.float32)
            u0, v0 = int(u), int(v)
            u1, v1 = min(u0 + 1, w_ - 1), min(v0 + 1, h_ - 1)
            du, dv = u - u0, v - v0
            q00 = img[v0, u0]
            q10 = img[v0, u1]
            q01 = img[v1, u0]
            q11 = img[v1, u1]
            return (
                (1 - du) * (1 - dv) * q00
                + du * (1 - dv) * q10
                + (1 - du) * dv * q01
                + du * dv * q11
            )

        failures = 0
        for r in range(h):
            for c in range(w):
                d = filtered[r, c]
                if not _np.isfinite(d) or d <= 0:
                    continue
                # 参照色（中心画素）
                ref_col = ref_img[r, c]
                # 3D点（参照カメラ座標）
                x_cam = (c - K[0, 2]) * d / K[0, 0]
                y_cam = (r - K[1, 2]) * d / K[1, 1]
                p_cam = _np.array([x_cam, y_cam, d], dtype=_np.float32)
                # ワールド座標
                p_world = R_ref.T @ (p_cam - T_ref)
                consistent = 0
                Hn, Wn, _ = neighbor_images[0].shape
                for i in range(neighbor_images.shape[0]):
                    R_src = neighbor_R[i]
                    T_src = neighbor_T[i]
                    # 投影
                    p_src = R_src @ p_world + T_src
                    if p_src[2] <= 0:
                        continue
                    u = K[0, 0] * p_src[0] / p_src[2] + K[0, 2]
                    v = K[1, 1] * p_src[1] / p_src[2] + K[1, 2]
                    if not (0 <= u < Wn and 0 <= v < Hn):
                        continue
                    nbr_col = _bilinear_color(neighbor_images[i], v, u)
                    # L2色差
                    diff = _np.linalg.norm(ref_col - nbr_col)
                    if diff < thresh:
                        consistent += 1
                if consistent < min_consistent:
                    filtered[r, c] = _np.nan
                    failures += 1

        _logging.info(
            f"{failures} points invalidated by photometric consistency check."
        )
        return filtered

    def filter_depth_map_by_geometric_consistency(
        self, ref_depth_map, ref_pose, neighbor_views_data, all_optimized_depths
    ):
        """
        幾何学的一貫性に基づくフィルタ（GPU実装と同等のロジック）
        各近傍ビューの深度と投影深度の相対誤差を評価し、一貫ビュー数が閾値未満なら無効化。
        90%超無効化された場合はフォールバックで元深度を返す。
        """
        import logging as _logging

        import numpy as _np

        _logging.info("Filtering depth map by geometric consistency...")
        if ref_depth_map is None:
            return ref_depth_map
        h, w = ref_depth_map.shape
        filtered = ref_depth_map.copy()
        initial_valid = int(_np.sum(_np.isfinite(ref_depth_map) & (ref_depth_map > 0)))

        K_ref = ref_pose["K"].astype(_np.float32)
        R_ref = ref_pose["R"].astype(_np.float32)
        T_ref = ref_pose["T"].astype(_np.float32)

        if abs(K_ref[0, 0]) < 1e-6 or abs(K_ref[1, 1]) < 1e-6:
            _logging.error(
                "Focal length is zero. Aborting geometric consistency check."
            )
            return ref_depth_map

        neighbor_K_list = []
        neighbor_R_list = []
        neighbor_T_list = []
        neighbor_depth_maps_list = []

        for view in neighbor_views_data:
            view_idx = view.get("image_idx")
            if view_idx in all_optimized_depths:
                neighbor_K_list.append(view["K"].astype(_np.float32))
                neighbor_R_list.append(view["R"].astype(_np.float32))
                neighbor_T_list.append(view["T"].astype(_np.float32))
                neighbor_depth_maps_list.append(
                    all_optimized_depths[view_idx].astype(_np.float32)
                )

        if not neighbor_depth_maps_list:
            _logging.warning(
                "No neighbor depth maps available for geometric consistency check."
            )
            return filtered

        neighbor_K_np = _np.stack(neighbor_K_list)
        neighbor_R_np = _np.stack(neighbor_R_list)
        neighbor_T_np = _np.stack(neighbor_T_list)
        neighbor_depth_maps_np = _np.stack(neighbor_depth_maps_list)

        thresh = float(
            getattr(self.config, "GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD", 0.05)
        )
        min_consistent = int(getattr(self.config, "GEOMETRIC_MIN_CONSISTENT_VIEWS", 2))

        failures = 0
        for r in range(h):
            for c in range(w):
                d_ref = filtered[r, c]
                if not _np.isfinite(d_ref) or d_ref <= 0:
                    continue
                # 3Dポイント（参照カメラ→世界）
                x_cam_ref = (c - K_ref[0, 2]) * d_ref / K_ref[0, 0]
                y_cam_ref = (r - K_ref[1, 2]) * d_ref / K_ref[1, 1]
                p_cam = _np.array([x_cam_ref, y_cam_ref, d_ref], dtype=_np.float32)
                p_world = R_ref.T @ (p_cam - T_ref)

                consistent = 0
                for i in range(neighbor_depth_maps_np.shape[0]):
                    K_src = neighbor_K_np[i]
                    R_src = neighbor_R_np[i]
                    T_src = neighbor_T_np[i]
                    depth_src = neighbor_depth_maps_np[i]
                    # 投影
                    p_src_cam = R_src @ p_world + T_src
                    p_src_h = K_src @ p_src_cam
                    d_proj = p_src_h[2]
                    if d_proj < 1e-6:
                        continue
                    u = p_src_h[0] / d_proj
                    v = p_src_h[1] / d_proj
                    Hn, Wn = depth_src.shape
                    if not (0 <= u < Wn and 0 <= v < Hn):
                        continue
                    rr, cc = int(round(v)), int(round(u))
                    if not (0 <= rr < Hn and 0 <= cc < Wn):
                        continue
                    d_actual = depth_src[rr, cc]
                    if not _np.isfinite(d_actual) or d_actual < 1e-6:
                        continue
                    rel_err = abs(d_proj - d_actual) / d_actual
                    if rel_err < thresh:
                        consistent += 1
                if consistent < min_consistent:
                    filtered[r, c] = _np.nan
                    failures += 1

        _logging.info(
            f"{failures} points ({(failures/(h*w))*100:.2f}%) invalidated by geometric consistency check."
        )
        if initial_valid > 0 and failures >= 0.9 * initial_valid:
            _logging.warning(
                "Geometric filter invalidated >90% of valid pixels. Returning unfiltered depth."
            )
            return ref_depth_map
        return filtered
