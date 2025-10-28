// kernels.cu - CuPy RawModule 用のPTX生成ソース
// 既存のCuPy実装と同等の簡易版Kernel群（checkerboard伝播、ランダムサーチ、
// 勾配コスト、投影ZNCCコスト、ACMH-lite選抜）

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

    // 4近傍
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

    depth[idx] = best_depth;
    cost[idx]  = best_cost;
}

// forward declaration for bilinear used below
__device__ __forceinline__ float bilinear(const float* img, int H, int W, float v, float u);

__device__ __forceinline__ float compute_cost_at(
    const float* ref_gray, const float* src_imgs,
    const float* K_ref, const float* R_ref, const float* T_ref,
    const float* src_K, const float* src_R, const float* src_T,
    const int H, const int W, const int N,
    const int r, const int c, const float d,
    const int topk, const int use_median, const int patch_size,
    const float sigma_color, const float zncc_eps, const float cov_required, const int min_valid,
    const float nx, const float ny, const float nz)
{
    if (!(d>0.0f)) return 1.0f;
    float fx = K_ref[0], fy = K_ref[4], cx = K_ref[2], cy = K_ref[5];
    // anchor at center
    float x0 = (((float)c) - cx) * d / fx;
    float y0 = (((float)r) - cy) * d / fy;
    float z0 = d;
    float NdotX0 = nx*x0 + ny*y0 + nz*z0;
    // R_ref^T rows packed column-major
    float Rrt00 = R_ref[0], Rrt01 = R_ref[3], Rrt02 = R_ref[6];
    float Rrt10 = R_ref[1], Rrt11 = R_ref[4], Rrt12 = R_ref[7];
    float Rrt20 = R_ref[2], Rrt21 = R_ref[5], Rrt22 = R_ref[8];

    float center_I = ref_gray[r*W + c];

    float diffs[32]; int dcnt = 0; int validViews = 0;
    for(int n=0;n<N;++n){
        const float* Kr = src_K + 9*n;
        const float* Rr = src_R + 9*n;
        const float* Tr = src_T + 3*n;
        const float* img = src_imgs + n*(H*W);
        int half = patch_size/2;
        float sumw = 0.0f;
        float sx0=0.0f, sy0=0.0f, sxx=0.0f, syy=0.0f, sxy=0.0f;
        int inside = 0; int total = (patch_size)*(patch_size);
        for(int pr=-half; pr<=half; ++pr){
            for(int pc=-half; pc<=half; ++pc){
                float wr = (float)(r+pr);
                float wc = (float)(c+pc);
                float I0 = 0.0f;
                if (wr>=0.0f && wr< (float)H && wc>=0.0f && wc<(float)W){
                    I0 = bilinear(ref_gray, H, W, wr, wc);
                }
                float u_ref = wc; float v_ref = wr;
                float denom = nx*((u_ref - cx)/fx) + ny*((v_ref - cy)/fy) + nz;
                if (fabsf(denom) < 1e-8f) continue;
                float z = NdotX0 / denom; if (!(z>0.0f)) continue;
                float x = (u_ref - cx) * z / fx;
                float y = (v_ref - cy) * z / fy;
                float px = x - T_ref[0];
                float py = y - T_ref[1];
                float pz = z - T_ref[2];
                float wx = Rrt00*px + Rrt01*py + Rrt02*pz;
                float wy = Rrt10*px + Rrt11*py + Rrt12*pz;
                float wz = Rrt20*px + Rrt21*py + Rrt22*pz;
                float sx = Rr[0]*wx + Rr[1]*wy + Rr[2]*wz + Tr[0];
                float sy = Rr[3]*wx + Rr[4]*wy + Rr[5]*wz + Tr[1];
                float sz = Rr[6]*wx + Rr[7]*wy + Rr[8]*wz + Tr[2];
                if (fabsf(sz) < 1e-6f) continue;
                float uu = Kr[0]*sx/sz + Kr[2];
                float vv = Kr[4]*sy/sz + Kr[5];
                float I1 = 0.0f;
                if (vv>=0.0f && vv < (float)H && uu>=0.0f && uu < (float)W){
                    I1 = bilinear(img, H, W, vv, uu);
                    inside += 1;
                }
                // adaptive weight by color difference from center
                float diffc = I0 - center_I;
                float w = expf(-(diffc*diffc) / (2.0f * (sigma_color*sigma_color) + 1e-6f));
                sumw += w;
                sx0 += w*I0; sy0 += w*I1;
                sxx += w*I0*I0; syy += w*I1*I1; sxy += w*I0*I1;
            }
        }
        if (sumw <= 1e-6f) continue;
        float coverage = (float)inside / (float)total;
        if (coverage < cov_required) continue;
        float mx = sx0/sumw, my = sy0/sumw;
        float vx = fmaxf(0.0f, sxx/sumw - mx*mx);
        float vy = fmaxf(0.0f, syy/sumw - my*my);
        float denomV = sqrtf(vx*vy);
        float zncc = (denomV > zncc_eps) ? ((sxy/sumw - mx*my) / denomV) : 0.0f;
        float costv = 0.5f * (1.0f - zncc);
        if (dcnt < 32){ diffs[dcnt++] = costv; validViews += 1; }
    }
    if (dcnt==0 || validViews < min_valid) return 1.0f;
    for(int i=0;i<dcnt;i++){
        int mi=i; float mv=diffs[i];
        for(int j=i+1;j<dcnt;j++){ if (diffs[j] < mv){ mi=j; mv=diffs[j]; } }
        float tmp=diffs[i]; diffs[i]=diffs[mi]; diffs[mi]=tmp;
    }
    int k = topk < dcnt ? topk : dcnt;
    if (use_median){ int mid = k/2; return diffs[mid]; }
    float acc=0.0f; for(int i=0;i<k;i++) acc += diffs[i];
    return acc / (float)k;
}

extern "C" __global__ void propagate_checker4_eval(
    float* depth, float* cost, float* normals, const unsigned char* mask,
    const int H, const int W, const int color, const int use8,
    const float* ref_gray, const float* src_imgs,
    const float* K_ref, const float* R_ref, const float* T_ref,
    const float* src_K, const float* src_R, const float* src_T,
    const int N, const int topk, const int use_median, const int patch_size,
    const float sigma_color, const float zncc_eps, const float cov_required, const int min_valid)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    if (((r+c)&1) != color) return;
    const int idx = r*W + c;
    float d_cur = depth[idx];
    float best_depth = d_cur;
    float nx0 = normals[3*idx+0];
    float ny0 = normals[3*idx+1];
    float nz0 = normals[3*idx+2];
    float best_cost  = compute_cost_at(ref_gray, src_imgs, K_ref, R_ref, T_ref, src_K, src_R, src_T,
                                       H, W, N, r, c, d_cur, topk, use_median, patch_size, sigma_color, zncc_eps, cov_required, min_valid, nx0, ny0, nz0);
    // neighbor candidates
    int rr, cc;
    #define TRY_NBR(R,C) \
        rr=(R); cc=(C); \
        if (rr>=0 && rr<H && cc>=0 && cc<W && mask[rr*W+cc]){ \
            float dnb = depth[rr*W+cc]; \
            if (dnb>0.0f){ \
                int nidx = rr*W+cc; \
                float nnx = normals[3*nidx+0]; \
                float nny = normals[3*nidx+1]; \
                float nnz = normals[3*nidx+2]; \
                float cn = compute_cost_at(ref_gray, src_imgs, K_ref, R_ref, T_ref, src_K, src_R, src_T, \
                                           H, W, N, r, c, dnb, topk, use_median, patch_size, sigma_color, zncc_eps, cov_required, min_valid, nnx, nny, nnz); \
                if (cn < best_cost){ best_cost = cn; best_depth = dnb; /* copy normal below */ } \
            } \
        }
    TRY_NBR(r-1,c); TRY_NBR(r+1,c); TRY_NBR(r,c-1); TRY_NBR(r,c+1);
    if (use8){ TRY_NBR(r-1,c-1); TRY_NBR(r-1,c+1); TRY_NBR(r+1,c-1); TRY_NBR(r+1,c+1); }
    #undef TRY_NBR
    if (best_depth != d_cur){
        int rr2, cc2;
        int drs[8] = {-1,1,0,0,-1,-1,1,1};
        int dcs[8] = {0,0,-1,1,-1,1,-1,1};
        int lim = use8?8:4;
        for(int k=0;k<lim;k++){
            rr2 = r + drs[k]; cc2 = c + dcs[k];
            if (rr2>=0 && rr2<H && cc2>=0 && cc2<W && mask[rr2*W+cc2]){
                int nidx2 = rr2*W+cc2;
                if (depth[nidx2] == best_depth){
                    normals[3*idx+0] = normals[3*nidx2+0];
                    normals[3*idx+1] = normals[3*nidx2+1];
                    normals[3*idx+2] = normals[3*nidx2+2];
                    break;
                }
            }
        }
    }
    depth[idx] = best_depth;
    cost[idx]  = best_cost;
}

extern "C" __global__ void random_search_depth_eval(
    float* depth, float* cost, float* normals, const unsigned char* mask,
    const int H, const int W, const float* depth_range,
    const float* ref_gray, const float* src_imgs,
    const float* K_ref, const float* R_ref, const float* T_ref,
    const float* src_K, const float* src_R, const float* src_T,
    const int N, const int topk, const int use_median, const int patch_size,
    const float sigma_color, const float zncc_eps, const int seed, const float dmin, const float dmax,
    const float cov_required, const int min_valid, const float normal_angle_deg, const float angle_decay)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r>=H || c>=W) return;
    if (!mask[r*W+c]) return;
    const int idx = r*W + c;
    float d = depth[idx];
    if (!(d>0.0f) || isnan(d) || isinf(d)) return;
    // per-pixel range
    float range = depth_range[idx];
    if (!(range>0.0f)) range = 0.0f;
    // xorshift
    unsigned int s = (unsigned int)(seed * 73856093u) ^ (unsigned int)(idx * 19349663u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float u = (float)(s & 0x00FFFFFF) * (1.0f / 16777215.0f);
    float delta = (u * 2.0f - 1.0f) * range;
    float dn = d + delta;
    if (dn < dmin) dn = d + fabsf(delta);
    if (dn > dmax) dn = d - fabsf(delta);
    if (dn <= dmin || dn >= dmax) return;
    // normal perturbation
    float nx = normals[3*idx+0];
    float ny = normals[3*idx+1];
    float nz = normals[3*idx+2];
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; float u1 = (float)(s & 0x00FFFFFF) * (1.0f/16777215.0f);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; float u2 = (float)((s>>1) & 0x00FFFFFF) * (1.0f/16777215.0f);
    float ax = u1*2.0f - 1.0f;
    float ay = u2*2.0f - 1.0f;
    float az = 1.0f - fabsf(ax) - fabsf(ay);
    float normA = sqrtf(ax*ax+ay*ay+az*az)+1e-6f; ax/=normA; ay/=normA; az/=normA;
    float angle = normal_angle_deg * (3.1415926535f/180.0f) * angle_decay * (u1*2.0f-1.0f);
    float ca = cosf(angle), sa = sinf(angle), one_c = 1.0f - ca;
    float dot = ax*nx+ay*ny+az*nz;
    float cx = ay*nz - az*ny;
    float cy = az*nx - ax*nz;
    float cz = ax*ny - ay*nx;
    float nnx = nx*ca + cx*sa + ax*dot*one_c;
    float nny = ny*ca + cy*sa + ay*dot*one_c;
    float nnz = nz*ca + cz*sa + az*dot*one_c;
    float nrm = sqrtf(nnx*nnx+nny*nny+nnz*nnz)+1e-6f; nnx/=nrm; nny/=nrm; nnz/=nrm;

    float cc = cost[idx];
    float cn = compute_cost_at(ref_gray, src_imgs, K_ref, R_ref, T_ref, src_K, src_R, src_T,
                               H, W, N, r, c, dn, topk, use_median, patch_size, sigma_color, zncc_eps, cov_required, min_valid, nnx, nny, nnz);
    if (cn < cc){ depth[idx] = dn; cost[idx] = cn; normals[3*idx+0]=nnx; normals[3*idx+1]=nny; normals[3*idx+2]=nnz; }
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

extern "C" __global__ void compute_cost_project(
    const float* ref_gray, const float* src_imgs,
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

    float fx = K_ref[0], fy = K_ref[4], cx = K_ref[2], cy = K_ref[5];
    float x = ( (float)c - cx ) * d / fx;
    float y = ( (float)r - cy ) * d / fy;
    float z = d;
    float px = x - T_ref[0];
    float py = y - T_ref[1];
    float pz = z - T_ref[2];
    float Rrt00 = R_ref[0], Rrt01 = R_ref[3], Rrt02 = R_ref[6];
    float Rrt10 = R_ref[1], Rrt11 = R_ref[4], Rrt12 = R_ref[7];
    float Rrt20 = R_ref[2], Rrt21 = R_ref[5], Rrt22 = R_ref[8];
    float wx = Rrt00*px + Rrt01*py + Rrt02*pz;
    float wy = Rrt10*px + Rrt11*py + Rrt12*pz;
    float wz = Rrt20*px + Rrt21*py + Rrt22*pz;

    float Iref = ref_gray[idx];
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
        int half = patch_size/2;
        float sumw = 0.0f;
        float sx0=0.0f, sy0=0.0f, sxx=0.0f, syy=0.0f, sxy=0.0f;
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
                float sx1 = Rr[0]*p0 + Rr[1]*p1 + Rr[2]*p2 + Tr[0];
                float sy1 = Rr[3]*p0 + Rr[4]*p1 + Rr[5]*p2 + Tr[1];
                float sz1 = Rr[6]*p0 + Rr[7]*p1 + Rr[8]*p2 + Tr[2];
                if (fabsf(sz1) < 1e-6f) continue;
                float uu = Kr[0]*sx1/sz1 + Kr[2];
                float vv = Kr[4]*sy1/sz1 + Kr[5];
                float I1 = bilinear(img, H, W, vv, uu);
                float w = 1.0f;
                sumw += w;
                sx0 += w*I0; sy0 += w*I1;
                sxx += w*I0*I0; syy += w*I1*I1; sxy += w*I0*I1;
            }
        }
        if (sumw <= 1e-6f) continue;
        float mx = sx0/sumw, my = sy0/sumw;
        float vx = fmaxf(0.0f, sxx/sumw - mx*mx);
        float vy = fmaxf(0.0f, syy/sumw - my*my);
        float denom = sqrtf(vx*vy);
        float zncc = (denom > zncc_eps) ? ((sxy/sumw - mx*my) / denom) : 0.0f;
        float costv = 0.5f * (1.0f - zncc);
        if (dcnt < 32){ diffs[dcnt++] = costv; }
    }
    if (dcnt==0){ cost[idx] = 1.0f; return; }
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
    for(int s=0;s<Hs;++s){
        float cs = costH[s*HW + idx];
        float ds = depthH[s*HW + idx];
        if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; }
        else if (cs < second_cost){ second_cost=cs; second_depth=ds; }
    }
    int rr, cc, nidx;
    rr=r-1; cc=c; if (rr>=0 && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
    rr=r+1; cc=c; if (rr<H && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
    rr=r; cc=c-1; if (cc>=0 && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
    rr=r; cc=c+1; if (cc<W && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
    if (use8){
        rr=r-1; cc=c-1; if (rr>=0 && cc>=0 && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
        rr=r-1; cc=c+1; if (rr>=0 && cc<W && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
        rr=r+1; cc=c-1; if (rr<H && cc>=0 && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
        rr=r+1; cc=c+1; if (rr<H && cc<W && mask[rr*W+cc]){ nidx=rr*W+cc; for(int s=0;s<Hs;++s){ float cs=costH[s*HW+nidx]; float ds=depthH[s*HW+nidx]; if (cs < best_cost){ second_cost=best_cost; second_depth=best_depth; best_cost=cs; best_depth=ds; } else if (cs < second_cost){ second_cost=cs; second_depth=ds; } } }
    }
    depthH[0*HW + idx] = best_depth; costH[0*HW + idx] = best_cost;
    if (Hs>1){ depthH[1*HW + idx] = second_depth; costH[1*HW + idx] = second_cost; }
}
