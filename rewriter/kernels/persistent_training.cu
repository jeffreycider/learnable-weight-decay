// Literal single-launch cooperative training interpreter.
// Device-resident stage scheduler; every stage and grid barrier executes inside
// this one cooperative launch. Host code constructs descriptors only.
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_bf16.h>
#ifndef CHISEL_ATTENTION_SHARED_STRIDE
#define CHISEL_ATTENTION_SHARED_STRIDE 128
#endif
#ifdef CHISEL_CUTLASS
#include "persistent_cutlass.cuh"
#endif
#ifdef CHISEL_CUTLASS_IEEE
#include "persistent_ieee_cutlass.cuh"
#endif
#ifdef CHISEL_CUTLASS_TF32X3
#include "persistent_tf32x3_cutlass.cuh"
#endif
#ifdef CHISEL_FP32_EMULATION
#include "persistent_fp32_emulation.cuh"
#endif
using u64 = unsigned long long;
using i64 = long long;
namespace cg = cooperative_groups;
using namespace nvcuda;

constexpr int WARPS=4;
constexpr int STRIDE=24;
enum { GEMM=0, REDUCE=1, BIAS_PARTIAL=2, ADAM=3, SOFTMAX=4, SOFTMAX_BACKWARD=5,
       LAYER_NORM=6, LAYER_NORM_BACKWARD=7, ADD=8, HEAD_LAYOUT=9,
       CONTEXT_CONCAT=10, CONTEXT_SPLIT=11, COPY=12, BROADCAST=13,
       TRAJECTORY_SCALE=14, TRAJECTORY_NORMALIZE=15, UNARY=16,
       RMS_NORM=17, RMS_NORM_BACKWARD=18, ROTARY=19,
       EMBEDDING=20, EMBEDDING_BACKWARD=21, ATTENTION_DELTA=22,
       KL=23, DOUBLE_REDUCE=24, PAIR_STATS=25, RADIUS=26, EDIT=27,
       GROUP_OBJECTIVE=28, OBJECTIVE_COMBINE=29, PARAMETER_COTANGENT=30,
       ROW_DIVIDE=31, RADIAL_COTANGENT=32, FACTOR_COTANGENT=33,
       CLIP_GRADIENT=34, FLASH_PROBABILITIES=35, STREAM_ATTENTION_FORWARD=36, STREAM_ATTENTION_DQ=37, STREAM_ATTENTION_DKDV=38, ATTENTION_ROW_DELTA=39, DOUBLE_POINTER_SUM=40, ZERO=41, EMBEDDING_SCATTER=42 };

__device__ float block_reduce(float value, float* scratch, bool maximum) {
    scratch[threadIdx.x]=value;
    __syncthreads();
    for(int span=blockDim.x/2;span;span/=2) {
        if(threadIdx.x<span) scratch[threadIdx.x]=maximum?
            fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+span]):
            scratch[threadIdx.x]+scratch[threadIdx.x+span];
        __syncthreads();
    }
    float result=scratch[0];
    __syncthreads();
    return result;
}
__device__ double block_reduce_double(double value, float* memory) {
    double* scratch=(double*)memory;scratch[threadIdx.x]=value;
    __syncthreads();
    for(int span=blockDim.x/2;span;span/=2) {
        if(threadIdx.x<span) scratch[threadIdx.x]+=scratch[threadIdx.x+span];
        __syncthreads();
    }
    double result=scratch[0];__syncthreads();return result;
}

__device__ __forceinline__ float tf32(float x) {
    unsigned int bits;
    asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x));
    return __uint_as_float(bits);
}
__device__ __forceinline__ float from_bits(u64 x) {
    return __uint_as_float((unsigned int)x);
}
__device__ __forceinline__ float bf16(float x) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    return __uint_as_float(((unsigned int)bits)<<16);
}

__device__ __forceinline__ void pack_components(float x,__nv_bfloat16* out,int index,int plane,bool residuals) {
    auto high=__float2bfloat16_rn(x);out[index]=high;
    if(residuals) {
        float remainder=__fsub_rn(x,__bfloat162float(high));
        auto middle=__float2bfloat16_rn(remainder);out[plane+index]=middle;
        out[2*plane+index]=__float2bfloat16_rn(__fsub_rn(remainder,__bfloat162float(middle)));
    }
}

__device__ void bf16_tensor_tile(const u64* s,int task,float* sh) {
    int lane=threadIdx.x%32,M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+15)/16,tm=(M+15)/16,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*16,c0=(tile%tn)*16;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float* c=(float*)s[3]+batch*s[17];
    __nv_bfloat16 *as=(__nv_bfloat16*)sh,*bs=(__nv_bfloat16*)(sh+256);float* out=sh+512;
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    if(s[21]==1 && c0>=r0+16)end=begin;
    if(s[21]==2)end=min(end,r0+16);
    if(s[21]==3)begin=max(begin,r0);
    for(int kk=begin;kk<end;kk+=16) {
        for(int i=lane;i<256;i+=32) {
            int ar=r0+i/16,ak=kk+i%16,bk=kk+i/16,bc=c0+i%16;
            as[i]=__float2bfloat16_rn(ar<M && ak<end?a[(flags&1)?ak*lda+ar:ar*lda+ak]:0.f);
            bs[i]=__float2bfloat16_rn(bk<end && bc<N?b[(flags&2)?bc*ldb+bk:bk*ldb+bc]:0.f);
        }
        __syncwarp();wmma::load_matrix_sync(fa,as,16);wmma::load_matrix_sync(fb,bs,16);
        wmma::mma_sync(acc,fa,fb,acc);__syncwarp();
    }
    wmma::store_matrix_sync(out,acc,16,wmma::mem_row_major);__syncwarp();
    for(int i=lane;i<256;i+=32) {
        int row=r0+i/16,col=c0+i%16;if(row<M && col<N) {
            float value=out[i];if(s[19]) value*=from_bits(s[19]);if((flags>>2)==1) value+=bias[col];
            c[(split*M+row)*N+col]=value;
        }
    }
}

// CTA-cooperative 32x64x32 tile. Four warps reuse the same A/B tiles;
// transposed operands are loaded in their contiguous physical direction.
__device__ void bf16_cta_tile(const u64* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+63)/64,tm=(M+31)/32,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*32,c0=(tile%tn)*64;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float* c=(float*)s[3]+batch*s[17];
    __nv_bfloat16 *as=(__nv_bfloat16*)sh,*bs=as+1024;float* out=sh+1536;
    int warp=threadIdx.x/32,wr=(warp/2)*16,wc=(warp%2)*32;
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb0,fb1;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc0,acc1;
    wmma::fill_fragment(acc0,0.f);wmma::fill_fragment(acc1,0.f);
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    if(s[21]==1 && c0>=r0+32)end=begin;
    if(s[21]==2)end=min(end,r0+32);
    if(s[21]==3)begin=max(begin,r0);
    for(int kk=begin;kk<end;kk+=32) {
        for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
            int ar=(flags&1)?i%32:i/32,ak=(flags&1)?i/32:i%32;
            as[ar*32+ak]=__float2bfloat16_rn(r0+ar<M && kk+ak<end?
                a[(flags&1)?(kk+ak)*lda+r0+ar:(r0+ar)*lda+kk+ak]:0.f);
        }
        for(int i=threadIdx.x;i<2048;i+=blockDim.x) {
            int bk=(flags&2)?i%32:i/64,bc=(flags&2)?i/32:i%64;
            bs[bk*64+bc]=__float2bfloat16_rn(kk+bk<end && c0+bc<N?
                b[(flags&2)?(c0+bc)*ldb+kk+bk:(kk+bk)*ldb+c0+bc]:0.f);
        }
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<32;ki+=16) {
            wmma::load_matrix_sync(fa,as+wr*32+ki,32);
            wmma::load_matrix_sync(fb0,bs+ki*64+wc,64);
            wmma::load_matrix_sync(fb1,bs+ki*64+wc+16,64);
            wmma::mma_sync(acc0,fa,fb0,acc0);wmma::mma_sync(acc1,fa,fb1,acc1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(out+wr*64+wc,acc0,64,wmma::mem_row_major);
    wmma::store_matrix_sync(out+wr*64+wc+16,acc1,64,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x) {
        int row=r0+i/64,col=c0+i%64;
        if(row<M && col<N) {
            float value=out[i];if(s[19]) value*=from_bits(s[19]);if((flags>>2)==1) value+=bias[col];
            c[(split*M+row)*N+col]=value;
        }
    }
}

// FP32 operands use a three-product TF32 decomposition at the same CTA tile.
__device__ void tf32_cta_tile(const u64* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+63)/64,tm=(M+31)/32,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*32,c0=(tile%tn)*64;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float *c=(float*)s[3]+batch*s[17],*aux=(float*)s[5];if(aux)aux+=batch*s[18];
    float *ah=sh,*al=ah+512,*bh=al+512,*bl=bh+1024,*out=bl+1024;
    int warp=threadIdx.x/32,wr=(warp/2)*16,wc=(warp%2)*32;
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> fa,fal;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> fb0,fb1,fbl0,fbl1;
    wmma::fragment<wmma::accumulator,16,16,8,float> acc0,acc1;
    wmma::fill_fragment(acc0,0.f);wmma::fill_fragment(acc1,0.f);
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    if(s[21]==1 && c0>=r0+32)end=begin;
    if(s[21]==2)end=min(end,r0+32);
    if(s[21]==3)begin=max(begin,r0);
    for(int kk=begin;kk<end;kk+=16) {
        for(int i=threadIdx.x;i<512;i+=blockDim.x) {
            int ar=(flags&1)?i%32:i/16,ak=(flags&1)?i/32:i%16;
            float value=r0+ar<M && kk+ak<end?a[(flags&1)?(kk+ak)*lda+r0+ar:(r0+ar)*lda+kk+ak]:0.f;
            ah[ar*16+ak]=tf32(value);al[ar*16+ak]=tf32(value-ah[ar*16+ak]);
        }
        for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
            int bk=(flags&2)?i%16:i/64,bc=(flags&2)?i/16:i%64;
            float value=kk+bk<end && c0+bc<N?b[(flags&2)?(c0+bc)*ldb+kk+bk:(kk+bk)*ldb+c0+bc]:0.f;
            bh[bk*64+bc]=tf32(value);bl[bk*64+bc]=tf32(value-bh[bk*64+bc]);
        }
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<16;ki+=8) {
            wmma::load_matrix_sync(fa,ah+wr*16+ki,16);wmma::load_matrix_sync(fal,al+wr*16+ki,16);
            wmma::load_matrix_sync(fb0,bh+ki*64+wc,64);wmma::load_matrix_sync(fbl0,bl+ki*64+wc,64);
            wmma::load_matrix_sync(fb1,bh+ki*64+wc+16,64);wmma::load_matrix_sync(fbl1,bl+ki*64+wc+16,64);
            wmma::mma_sync(acc0,fa,fb0,acc0);wmma::mma_sync(acc0,fal,fb0,acc0);wmma::mma_sync(acc0,fa,fbl0,acc0);
            wmma::mma_sync(acc1,fa,fb1,acc1);wmma::mma_sync(acc1,fal,fb1,acc1);wmma::mma_sync(acc1,fa,fbl1,acc1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(out+wr*64+wc,acc0,64,wmma::mem_row_major);
    wmma::store_matrix_sync(out+wr*64+wc+16,acc1,64,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x) {
        int row=r0+i/64,col=c0+i%64;
        if(row<M && col<N) {
            float value=out[i];if(s[19]) value*=from_bits(s[19]);int epilogue=flags>>2;
            if(epilogue==1 || epilogue==2)value+=bias[col];
            if(epilogue==2){aux[row*N+col]=value;value=0.5f*value*(1.f+erff(value*0.7071067811865475f));}
            else if(epilogue==3){float z=aux[row*N+col];value*=0.5f*(1.f+erff(z*0.7071067811865475f))+z*0.3989422804014327f*expf(-0.5f*z*z);}
            c[(split*M+row)*N+col]=value;
        }
    }
}

__device__ void bf16_big_tile(const u64* s,int task,float* sh) {
    constexpr int TM=64,TN=128,TK=32;
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11],splits=s[12],chunk=s[13];
    int tn=(N+TN-1)/TN,tm=(M+TM-1)/TM,per=tm*tn*splits,batch=task/per;task%=per;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=tile/tn*TM,c0=tile%tn*TN;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];float* c=(float*)s[3]+batch*s[17];
    __nv_bfloat16 *as=(__nv_bfloat16*)sh,*bs=as+TM*TK;int warp=threadIdx.x/32,wr=warp/2*32,wc=warp%2*64;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][4];
    #pragma unroll
    for(int r=0;r<2;r++){
        #pragma unroll
        for(int col=0;col<4;col++)wmma::fill_fragment(acc[r][col],0.f);
    }
    int begin=split*chunk,end=min(K,(split+1)*chunk);if(s[21]==1 && c0>=r0+TM)end=begin;if(s[21]==2)end=min(end,r0+TM);if(s[21]==3)begin=max(begin,r0);
    for(int kk=begin;kk<end;kk+=TK){
        for(int i=threadIdx.x;i<TM*TK;i+=blockDim.x){int r=(flags&1)?i%TM:i/TK,k=(flags&1)?i/TM:i%TK;as[r*TK+k]=__float2bfloat16_rn(r0+r<M && kk+k<end?a[(flags&1)?(kk+k)*lda+r0+r:(r0+r)*lda+kk+k]:0.f);}
        for(int i=threadIdx.x;i<TK*TN;i+=blockDim.x){int k=(flags&2)?i%TK:i/TN,col=(flags&2)?i/TK:i%TN;bs[k*TN+col]=__float2bfloat16_rn(kk+k<end && c0+col<N?b[(flags&2)?(c0+col)*ldb+kk+k:(kk+k)*ldb+c0+col]:0.f);}
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<TK;ki+=16){
            #pragma unroll
            for(int r=0;r<2;r++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;wmma::load_matrix_sync(fa,as+(wr+r*16)*TK+ki,TK);
                #pragma unroll
                for(int col=0;col<4;col++){
                    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fb;wmma::load_matrix_sync(fb,bs+ki*TN+wc+col*16,TN);wmma::mma_sync(acc[r][col],fa,fb,acc[r][col]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<2;r++){
        #pragma unroll
        for(int col=0;col<4;col++)wmma::store_matrix_sync(sh+(wr+r*16)*TN+wc+col*16,acc[r][col],TN,wmma::mem_row_major);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<TM*TN;i+=blockDim.x){int r=r0+i/TN,col=c0+i%TN;if(r<M && col<N){float value=sh[i];if(s[19])value*=from_bits(s[19]);if((flags>>2)==1)value+=bias[col];c[(split*M+r)*N+col]=value;}}
}

__device__ void tensor_tile(const u64* s, int task, float* sh, float* shared_a, bool share_a) {
    int lane=threadIdx.x%32;
    int M=s[6], N=s[7], K=s[8], lda=s[9], ldb=s[10];
    int flags=s[11], splits=s[12], chunk=s[13];
    int tiles_n=(N+15)/16, tiles_m=(M+15)/16;
    int tasks_per_batch=tiles_m*tiles_n*splits;
    int batch=task/tasks_per_batch;
    task%=tasks_per_batch;
    int split=task/(tiles_m*tiles_n), tile=task%(tiles_m*tiles_n);
    int row0=(tile/tiles_n)*16, col0=(tile%tiles_n)*16;
    const float* a=(const float*)s[1]; const float* b=(const float*)s[2];
    float* c=(float*)s[3]; const float* bias=(const float*)s[4];
    float* aux=(float*)s[5];
    a+=batch*s[15]; b+=batch*s[16]; c+=batch*s[17];
    if(aux) aux+=batch*s[18];
    float *ah=share_a?shared_a:sh, *al=ah+128, *bh=sh+256, *bl=sh+384, *out=sh+512;
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> fa, fal;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> fb, fbl;
    wmma::fragment<wmma::accumulator,16,16,8,float> accumulator;
    wmma::fill_fragment(accumulator,0.f);
    int begin=split*chunk, end=min(K,begin+chunk);
    if(s[21]==1 && col0>=row0+16)end=begin;
    if(s[21]==2)end=min(end,row0+16);
    if(s[21]==3)begin=max(begin,row0);
    for(int kk=begin;kk<end;kk+=8) {
        for(int i=share_a?threadIdx.x:lane;i<128;i+=share_a?blockDim.x:32) {
            int ar=row0+i/8, ak=kk+i%8;
            float av=(ar<M && ak<end)?a[(flags&1)?ak*lda+ar:ar*lda+ak]:0.f;
            ah[i]=tf32(av); al[i]=tf32(av-ah[i]);
        }
        for(int i=lane;i<128;i+=32) {
            int bk=kk+i/16, bc=col0+i%16;
            float bv=(bk<end && bc<N)?b[(flags&2)?bc*ldb+bk:bk*ldb+bc]:0.f;
            bh[i]=tf32(bv); bl[i]=tf32(bv-bh[i]);
        }
        if(share_a) __syncthreads(); else __syncwarp();
        wmma::load_matrix_sync(fa,ah,8); wmma::load_matrix_sync(fal,al,8);
        wmma::load_matrix_sync(fb,bh,16); wmma::load_matrix_sync(fbl,bl,16);
        // TF32x3 retains the FP32 residual products; no BF16 actor conversion.
        wmma::mma_sync(accumulator,fal,fb,accumulator);
        wmma::mma_sync(accumulator,fa,fbl,accumulator);
        wmma::mma_sync(accumulator,fa,fb,accumulator);
        if(share_a) __syncthreads(); else __syncwarp();
    }
    wmma::store_matrix_sync(out,accumulator,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<256;i+=32) {
        int row=row0+i/16, col=col0+i%16;
        if(row<M && col<N) {
            float v=out[i]; int epilogue=flags>>2;
            if(s[19]) v*=from_bits(s[19]);
            if(epilogue==1 || epilogue==2) v+=bias[col];
            if(epilogue==2) {
                aux[row*N+col]=v;
                v=0.5f*v*(1.f+erff(v*0.7071067811865475f));
            } else if(epilogue==3) {
                float z=aux[row*N+col];
                v*=0.5f*(1.f+erff(z*0.7071067811865475f))
                    +z*0.3989422804014327f*expf(-0.5f*z*z);
            }
            c[(split*M+row)*N+col]=v;
        }
    }
}

// IEEE FP32 forward primitive. No TF32/BF16 operand conversion. Each thread
// accumulates a4x8 register tile using explicit FP32 FMAs, sharing64x16/16x64
// operand tiles across the CTA. This is deliberately separate from TF32x3:
// forward finite-edit sensitivity must not inherit tensor-core quantization.
__device__ void ieee_cta_tile(const u64* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+63)/64,tm=(M+63)/64,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*64,c0=(tile%tn)*64;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float *c=(float*)s[3]+batch*s[17],*aux=(float*)s[5];if(aux)aux+=batch*s[18];
    float *as=sh,*bs=sh+1024;
    int tr=threadIdx.x/8,tc=threadIdx.x%8;
    float accum[4][8];
    #pragma unroll
    for(int i=0;i<4;i++) {
        #pragma unroll
        for(int j=0;j<8;j++)accum[i][j]=0.f;
    }
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    if(s[21]==1 && c0>=r0+64)end=begin;
    if(s[21]==2)end=min(end,r0+64);
    if(s[21]==3)begin=max(begin,r0);
    for(int kk=begin;kk<end;kk+=16) {
        for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
            int ar=(flags&1)?i%64:i/16,ak=(flags&1)?i/64:i%16;
            as[ar*16+ak]=(r0+ar<M && kk+ak<end)?a[(flags&1)?(kk+ak)*lda+r0+ar:(r0+ar)*lda+kk+ak]:0.f;
            int bk=(flags&2)?i%16:i/64,bc=(flags&2)?i/16:i%64;
            bs[bk*64+bc]=(kk+bk<end && c0+bc<N)?b[(flags&2)?(c0+bc)*ldb+kk+bk:(kk+bk)*ldb+c0+bc]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;k++) {
            float av[4],bv[8];
            #pragma unroll
            for(int i=0;i<4;i++)av[i]=as[(tr+i*16)*16+k];
            #pragma unroll
            for(int j=0;j<8;j++)bv[j]=bs[k*64+tc+j*8];
            #pragma unroll
            for(int i=0;i<4;i++) {
                #pragma unroll
                for(int j=0;j<8;j++)accum[i][j]=fmaf(av[i],bv[j],accum[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++) {
        #pragma unroll
        for(int j=0;j<8;j++) {
            int row=r0+tr+i*16,col=c0+tc+j*8;
            if(row<M && col<N) {
                float v=accum[i][j];int epilogue=flags>>2;
                if(s[19])v*=from_bits(s[19]);
                if(epilogue==1 || epilogue==2)v+=bias[col];
                if(epilogue==2) {aux[row*N+col]=v;v=.5f*v*(1.f+erff(v*.7071067811865475f));}
                else if(epilogue==3) {float z=aux[row*N+col];v*=.5f*(1.f+erff(z*.7071067811865475f))+z*.3989422804014327f*expf(-.5f*z*z);}
                c[(split*M+row)*N+col]=v;
            }
        }
    }
}

// Rank-eight anchored contractions otherwise waste seven eighths of the
// 64-column IEEE tile. Keep the same sequential FP32 FMA accumulation.
__device__ void ieee_skinny_tile(const u64* s,int task,float* sh) {
#ifndef CHISEL_IEEE_SKINNY_K
#define CHISEL_IEEE_SKINNY_K 32
#endif
    constexpr int SK=CHISEL_IEEE_SKINNY_K;
    static_assert(64*(SK+1)+SK*8<=4*2816,"skinny tile exceeds workspace");
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+7)/8,tm=(M+63)/64,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*64,c0=(tile%tn)*8;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float *c=(float*)s[3]+batch*s[17],*aux=(float*)s[5];if(aux)aux+=batch*s[18];
    float *as=sh,*bs=sh+64*(SK+1);
    int tr=threadIdx.x/8,tc=threadIdx.x%8;
    float accum[4]={0.f,0.f,0.f,0.f};
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    for(int kk=begin;kk<end;kk+=SK) {
        for(int i=threadIdx.x;i<64*SK;i+=blockDim.x) {
            int ar=(flags&1)?i%64:i/SK,ak=(flags&1)?i/64:i%SK;
            as[ar*(SK+1)+ak]=(r0+ar<M && kk+ak<end)?a[(flags&1)?(kk+ak)*lda+r0+ar:(r0+ar)*lda+kk+ak]:0.f;
        }
        for(int i=threadIdx.x;i<SK*8;i+=blockDim.x) {
            int bk=(flags&2)?i%SK:i/8,bc=(flags&2)?i/SK:i%8;
            bs[bk*8+bc]=(kk+bk<end && c0+bc<N)?b[(flags&2)?(c0+bc)*ldb+kk+bk:(kk+bk)*ldb+c0+bc]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<SK;k++) {
            float bv=bs[k*8+tc];
            #pragma unroll
            for(int i=0;i<4;i++)accum[i]=fmaf(as[(tr+i*16)*(SK+1)+k],bv,accum[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++) {
        int row=r0+tr+i*16,col=c0+tc;
        if(row<M && col<N) {
            float v=accum[i];int epilogue=flags>>2;
            if(s[19])v*=from_bits(s[19]);
            if(epilogue==1 || epilogue==2)v+=bias[col];
            if(epilogue==2){aux[row*N+col]=v;v=.5f*v*(1.f+erff(v*.7071067811865475f));}
            else if(epilogue==3){float z=aux[row*N+col];v*=.5f*(1.f+erff(z*.7071067811865475f))+z*.3989422804014327f*expf(-.5f*z*z);}
            c[(split*M+row)*N+col]=v;
        }
    }
}

// Optional four-way K reduction for rank-eight contractions. Every operand
// remains FP32; the changed reduction order requires gradient qualification.
__device__ void ieee_skinny_parallel_tile(const u64* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],lda=s[9],ldb=s[10],flags=s[11];
    int splits=s[12],chunk=s[13],tn=(N+7)/8,tm=(M+63)/64,per_batch=tm*tn*splits;
    int batch=task/per_batch;task%=per_batch;
    int split=task/(tm*tn),tile=task%(tm*tn),r0=(tile/tn)*64,c0=(tile%tn)*8;
    const float *a=(const float*)s[1]+batch*s[15],*b=(const float*)s[2]+batch*s[16],*bias=(const float*)s[4];
    float *c=(float*)s[3]+batch*s[17],*aux=(float*)s[5];if(aux)aux+=batch*s[18];
    float *as=sh,*bs=sh+64*33;
    int warp=threadIdx.x/32,tc=threadIdx.x%8,kr=(threadIdx.x%32)/8;
    float accum[16]={};
    int begin=split*chunk,end=min(K,(split+1)*chunk);
    for(int kk=begin;kk<end;kk+=32) {
        for(int i=threadIdx.x;i<2048;i+=blockDim.x) {
            int ar=(flags&1)?i%64:i/32,ak=(flags&1)?i/64:i%32;
            as[ar*33+ak]=(r0+ar<M && kk+ak<end)?a[(flags&1)?(kk+ak)*lda+r0+ar:(r0+ar)*lda+kk+ak]:0.f;
        }
        for(int i=threadIdx.x;i<256;i+=blockDim.x) {
            int bk=(flags&2)?i%32:i/8,bc=(flags&2)?i/32:i%8;
            bs[bk*8+bc]=(kk+bk<end && c0+bc<N)?b[(flags&2)?(c0+bc)*ldb+kk+bk:(kk+bk)*ldb+c0+bc]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int k=kr;k<32;k+=4) {
            float bv=bs[k*8+tc];
            #pragma unroll
            for(int i=0;i<16;i++)accum[i]=fmaf(as[(warp*16+i)*33+k],bv,accum[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<16;i++) {
        float v=accum[i];
        v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,8));
        v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,16));
        int row=r0+warp*16+i,col=c0+tc;
        if(kr==0 && row<M && col<N) {int epilogue=flags>>2;
            if(s[19])v*=from_bits(s[19]);
            if(epilogue==1 || epilogue==2)v+=bias[col];
            if(epilogue==2){aux[row*N+col]=v;v=.5f*v*(1.f+erff(v*.7071067811865475f));}
            else if(epilogue==3){float z=aux[row*N+col];v*=.5f*(1.f+erff(z*.7071067811865475f))+z*.3989422804014327f*expf(-.5f*z*z);}
            c[(split*M+row)*N+col]=v;
        }
    }
}

template<bool Packed>
__device__ const float* attention_pointer(u64 pointer,int offset) {
    return (const float*)((const char*)pointer+offset*(Packed?2:4));
}
template<bool Packed>
__device__ void load_attention_tile(__nv_bfloat16* target,const float* source,int offset,int count,int valid) {
    if constexpr(Packed) {
        const auto* packed=(const __nv_bfloat16*)source+offset;
        for(int i=threadIdx.x*8;i<count;i+=blockDim.x*8){
            unsigned shared=(unsigned)__cvta_generic_to_shared(target+(i/128)*CHISEL_ATTENTION_SHARED_STRIDE+i%128);
            int bytes=i<valid?16:0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(shared),"l"(packed+i),"r"(bytes));
        }
        asm volatile("cp.async.commit_group;");asm volatile("cp.async.wait_group 0;");
    }else{
        for(int i=threadIdx.x;i<count;i+=blockDim.x)target[i]=__float2bfloat16_rn(i<valid?source[offset+i]:0.f);
    }
}

// Flash-style online attention. One CTA owns16queries of one128-widehead.
// No nested launch or device library call; all work stays inside interpreter.
template<bool Packed>
__device__ void attention_forward_tile(const u64* s,int task,float* sh) {
    constexpr int AD=Packed?CHISEL_ATTENTION_SHARED_STRIDE:128;
    int T=s[6],qt=(T+15)/16,h=task/qt,q0=(task%qt)*16;
    const float *q=attention_pointer<Packed>(s[1],h*T*128),*k=attention_pointer<Packed>(s[2],h*T*128),*v=attention_pointer<Packed>(s[3],h*T*128);
    float *out=(float*)s[4]+h*T*128,*scores=(float*)s[5]+h*T*T;
    float scale=from_bits(s[8]);
    __nv_bfloat16 *qs=(__nv_bfloat16*)sh,*kv=(__nv_bfloat16*)(sh+8*AD),*ps=(__nv_bfloat16*)(sh+40*AD+1024);
    float *ss=sh+40*AD,*os=sh+40*AD+1536,*ms=sh+40*AD+3584,*ls=ms+16,*alphas=ls+16;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wc=warp*32;
    load_attention_tile<Packed>(qs,q,q0*128,2048,max(0,min(T-q0,16))*128);
    if(threadIdx.x<16){ms[threadIdx.x]=__uint_as_float(0xff800000u);ls[threadIdx.x]=0.f;}
    wmma::fragment<wmma::accumulator,16,16,16,float> oa,ob;
    wmma::fill_fragment(oa,0.f);wmma::fill_fragment(ob,0.f);
    __syncthreads();
    for(int k0=((min(q0+16,T)-1)/64)*64;k0>=0;k0-=64) {
        load_attention_tile<Packed>(kv,k,k0*128,8192,max(0,min(T-k0,64))*128);
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qa;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> kb;
        wmma::fragment<wmma::accumulator,16,16,16,float> dot;wmma::fill_fragment(dot,0.f);
        #pragma unroll
        for(int d=0;d<128;d+=16){wmma::load_matrix_sync(qa,qs+d,AD);wmma::load_matrix_sync(kb,kv+warp*16*AD+d,AD);wmma::mma_sync(dot,qa,kb,dot);}
        wmma::store_matrix_sync(ss+warp*16,dot,64,wmma::mem_row_major);
        wmma::store_matrix_sync(os+wc,oa,128,wmma::mem_row_major);wmma::store_matrix_sync(os+wc+16,ob,128,wmma::mem_row_major);
        __syncthreads();
        for(int row=warp;row<16;row+=4) {
            float maximum=__uint_as_float(0xff800000u);
            for(int col=lane;col<64;col+=32)if(q0+row<T && k0+col<=q0+row)maximum=fmaxf(maximum,ss[row*64+col]*scale);
            for(int delta=16;delta;delta/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,delta));
            float newer=fmaxf(ms[row],maximum),alpha=isfinite(ms[row])?expf(ms[row]-newer):0.f,sum=0.f;
            if(!isfinite(newer))newer=0.f;
            for(int col=lane;col<64;col+=32) {
                bool valid=q0+row<T && k0+col<T && k0+col<=q0+row;
                float score=ss[row*64+col]*scale,p=valid?expf(score-newer):0.f;
                ps[row*64+col]=__float2bfloat16_rn(p);sum+=p;
                if(!s[9] && q0+row<T && k0+col<T)scores[(q0+row)*T+k0+col]=score;
            }
            for(int delta=16;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta);
            if(lane==0){ls[row]=ls[row]*alpha+sum;ms[row]=newer;alphas[row]=alpha;}
        }
        __syncthreads();
        for(int i=threadIdx.x;i<2048;i+=blockDim.x)os[i]*=alphas[i/128];
        load_attention_tile<Packed>(kv,v,k0*128,8192,max(0,min(T-k0,64))*128);
        __syncthreads();
        wmma::load_matrix_sync(oa,os+wc,128,wmma::mem_row_major);wmma::load_matrix_sync(ob,os+wc+16,128,wmma::mem_row_major);
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> va,vb;
        #pragma unroll
        for(int ki=0;ki<64;ki+=16){
            wmma::load_matrix_sync(pa,ps+ki,64);wmma::load_matrix_sync(va,kv+ki*AD+wc,AD);wmma::load_matrix_sync(vb,kv+ki*AD+wc+16,AD);
            wmma::mma_sync(oa,pa,va,oa);wmma::mma_sync(ob,pa,vb,ob);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(os+wc,oa,128,wmma::mem_row_major);wmma::store_matrix_sync(os+wc+16,ob,128,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x)if(q0+i/128<T)out[(q0+i/128)*128+i%128]=bf16(os[i]/ls[i/128]);
    if(s[9] && threadIdx.x<16 && q0+threadIdx.x<T)((float*)s[5])[h*T+q0+threadIdx.x]=ms[threadIdx.x]+logf(ls[threadIdx.x]);
}

template<bool Packed>
__device__ void attention_dq_tile(const u64* s,int task,float* sh) {
    constexpr int AD=Packed?CHISEL_ATTENTION_SHARED_STRIDE:128;
    int T=s[6],qt=(T+15)/16,h=task/qt,q0=(task%qt)*16;
    const float *dy=attention_pointer<Packed>(s[1],h*T*128),*v=attention_pointer<Packed>(s[2],h*T*128),*k=attention_pointer<Packed>(s[3],h*T*128);
    const float *p=(const float*)s[4]+h*T*T,*delta=(const float*)s[5]+h*T;
    float* dq=(float*)s[9]+h*T*128;float scale=from_bits(s[8]);
    __nv_bfloat16 *dys=(__nv_bfloat16*)sh,*kv=(__nv_bfloat16*)(sh+8*AD),*ds=(__nv_bfloat16*)(sh+40*AD+1024);
    float* dp=sh+40*AD;int warp=threadIdx.x/32,wc=warp*32;
    __nv_bfloat16* qs=(__nv_bfloat16*)(sh+40*AD+1536);float* qks=sh+48*AD+1536;
    load_attention_tile<Packed>(dys,dy,q0*128,2048,max(0,min(T-q0,16))*128);
    if(s[11])load_attention_tile<Packed>(qs,(const float*)s[10],(h*T+q0)*128,2048,max(0,min(T-q0,16))*128);
    wmma::fragment<wmma::accumulator,16,16,16,float> da,db;wmma::fill_fragment(da,0.f);wmma::fill_fragment(db,0.f);
    __syncthreads();
    for(int k0=0;k0<min(q0+16,T);k0+=64) {
        load_attention_tile<Packed>(kv,v,k0*128,8192,max(0,min(T-k0,64))*128);
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> ya;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> vb;
        wmma::fragment<wmma::accumulator,16,16,16,float> dot;wmma::fill_fragment(dot,0.f);
        #pragma unroll
        for(int d=0;d<128;d+=16){wmma::load_matrix_sync(ya,dys+d,AD);wmma::load_matrix_sync(vb,kv+warp*16*AD+d,AD);wmma::mma_sync(dot,ya,vb,dot);}
        wmma::store_matrix_sync(dp+warp*16,dot,64,wmma::mem_row_major);__syncthreads();
        load_attention_tile<Packed>(kv,k,k0*128,8192,max(0,min(T-k0,64))*128);
        __syncthreads();
        if(s[11]) {
            wmma::fill_fragment(dot,0.f);
            #pragma unroll
            for(int d=0;d<128;d+=16){wmma::load_matrix_sync(ya,qs+d,AD);wmma::load_matrix_sync(vb,kv+warp*16*AD+d,AD);wmma::mma_sync(dot,ya,vb,dot);}
            wmma::store_matrix_sync(qks+warp*16,dot,64,wmma::mem_row_major);__syncthreads();
        }
        for(int i=threadIdx.x;i<1024;i+=blockDim.x){int row=q0+i/64,col=k0+i%64;
            float prob=row<T && col<T && col<=row?(s[11]?expf(qks[i]*scale-((const float*)s[4])[h*T+row]):p[row*T+col]):0.f;
            ds[i]=__float2bfloat16_rn(row<T?prob*(dp[i]-delta[row])*scale:0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> sa;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> ka,kb;
        #pragma unroll
        for(int ki=0;ki<64;ki+=16){wmma::load_matrix_sync(sa,ds+ki,64);wmma::load_matrix_sync(ka,kv+ki*AD+wc,AD);wmma::load_matrix_sync(kb,kv+ki*AD+wc+16,AD);wmma::mma_sync(da,sa,ka,da);wmma::mma_sync(db,sa,kb,db);}
        __syncthreads();
    }
    float* out=sh+8*AD;wmma::store_matrix_sync(out+wc,da,128,wmma::mem_row_major);wmma::store_matrix_sync(out+wc+16,db,128,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x)if(q0+i/128<T)dq[(q0+i/128)*128+i%128]=bf16(out[i]);
}

template<bool Packed>
__device__ void attention_dkdv_tile(const u64* s,int task,float* sh) {
    constexpr int AD=Packed?CHISEL_ATTENTION_SHARED_STRIDE:128;
    int T=s[6],kt=(T+15)/16,h=task/kt,k0=(task%kt)*16;
    const float *dy=attention_pointer<Packed>(s[1],h*T*128),*v=attention_pointer<Packed>(s[2],h*T*128),*q=attention_pointer<Packed>(s[3],h*T*128);
    const float *p=(const float*)s[4]+h*T*T,*delta=(const float*)s[5]+h*T;
    float *dk=(float*)s[9]+h*T*128,*dv=(float*)s[10]+h*T*128;float scale=from_bits(s[8]);
    __nv_bfloat16 *ys=(__nv_bfloat16*)sh,*qvs=(__nv_bfloat16*)(sh+32*AD),*ds=(__nv_bfloat16*)(sh+64*AD+1024),*ps=(__nv_bfloat16*)(sh+64*AD+1536);
    float* dp=sh+64*AD;int warp=threadIdx.x/32,wc=warp*32;
    __nv_bfloat16* ks=(__nv_bfloat16*)(sh+64*AD+2048);
    if(s[12])load_attention_tile<Packed>(ks,(const float*)s[11],(h*T+k0)*128,2048,max(0,min(T-k0,16))*128);
    wmma::fragment<wmma::accumulator,16,16,16,float> ka,kb,va,vb;
    wmma::fill_fragment(ka,0.f);wmma::fill_fragment(kb,0.f);wmma::fill_fragment(va,0.f);wmma::fill_fragment(vb,0.f);
    for(int q0=(k0/64)*64;q0<T;q0+=64) {
        load_attention_tile<Packed>(ys,dy,q0*128,8192,max(0,min(T-q0,64))*128);
        load_attention_tile<Packed>(qvs,v,k0*128,2048,max(0,min(T-k0,16))*128);
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> ya;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> vt;
        wmma::fragment<wmma::accumulator,16,16,16,float> dot;wmma::fill_fragment(dot,0.f);
        #pragma unroll
        for(int d=0;d<128;d+=16){wmma::load_matrix_sync(ya,ys+warp*16*AD+d,AD);wmma::load_matrix_sync(vt,qvs+d,AD);wmma::mma_sync(dot,ya,vt,dot);}
        wmma::store_matrix_sync(dp+warp*16*16,dot,16,wmma::mem_row_major);__syncthreads();
        load_attention_tile<Packed>(qvs,q,q0*128,8192,max(0,min(T-q0,64))*128);
        __syncthreads();
        if(s[12]){
            wmma::fill_fragment(dot,0.f);
            #pragma unroll
            for(int d=0;d<128;d+=16){wmma::load_matrix_sync(ya,qvs+warp*16*AD+d,AD);wmma::load_matrix_sync(vt,ks+d,AD);wmma::mma_sync(dot,ya,vt,dot);}
            wmma::store_matrix_sync(sh+64*AD+1024+warp*16*16,dot,16,wmma::mem_row_major);__syncthreads();
        }
        float pv[8],sv[8];
        #pragma unroll
        for(int j=0;j<8;j++){int i=threadIdx.x+j*128,row=q0+i/16,col=k0+i%16;
            float prob=row<T && col<T && col<=row?(s[12]?expf(sh[64*AD+1024+i]*scale-((const float*)s[4])[h*T+row]):p[row*T+col]):0.f;
            pv[j]=prob;sv[j]=row<T?prob*(dp[i]-delta[row])*scale:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<8;j++){int i=threadIdx.x+j*128;ps[i]=__float2bfloat16_rn(pv[j]);ds[i]=__float2bfloat16_rn(sv[j]);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> sa,pa;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> qa,qb,ya0,yb0;
        #pragma unroll
        for(int qi=0;qi<64;qi+=16){
            wmma::load_matrix_sync(sa,ds+qi*16,16);wmma::load_matrix_sync(pa,ps+qi*16,16);
            wmma::load_matrix_sync(qa,qvs+qi*AD+wc,AD);wmma::load_matrix_sync(qb,qvs+qi*AD+wc+16,AD);
            wmma::load_matrix_sync(ya0,ys+qi*AD+wc,AD);wmma::load_matrix_sync(yb0,ys+qi*AD+wc+16,AD);
            wmma::mma_sync(ka,sa,qa,ka);wmma::mma_sync(kb,sa,qb,kb);wmma::mma_sync(va,pa,ya0,va);wmma::mma_sync(vb,pa,yb0,vb);
        }
        __syncthreads();
    }
    float *ok=sh,*ov=sh+4096;
    wmma::store_matrix_sync(ok+wc,ka,128,wmma::mem_row_major);wmma::store_matrix_sync(ok+wc+16,kb,128,wmma::mem_row_major);
    wmma::store_matrix_sync(ov+wc,va,128,wmma::mem_row_major);wmma::store_matrix_sync(ov+wc+16,vb,128,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x)if(k0+i/128<T){dk[(k0+i/128)*128+i%128]=bf16(ok[i]);dv[(k0+i/128)*128+i%128]=bf16(ov[i]);}
}

extern "C" __global__ void persistent_training(const u64* stages, int count,
                                                int* cursor, i64* step, u64* clocks) {
    cg::grid_group grid=cg::this_grid();
    int global=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=gridDim.x*blockDim.x;
#ifdef CHISEL_DYNAMIC_SHARED_BYTES
    extern __shared__ __align__(32) float dynamic_workspace[];
    auto* workspace=reinterpret_cast<float (*)[CHISEL_ATTENTION_SHARED_STRIDE==136?3008:2816]>(dynamic_workspace);
#else
    __shared__ __align__(32) float workspace[WARPS][CHISEL_ATTENTION_SHARED_STRIDE==136?3008:2816];
#endif
    __shared__ int first_task;
    if(global==0) { ++*step; *cursor=0; clocks[0]=clock64(); }
    grid.sync();
    for(int stage=0;stage<count;stage++) {
        const u64* s=stages+stage*STRIDE;
        int op=s[0];
        if(op==43 || op==49) {
            const float* x=(const float*)s[1];auto* out=(__nv_bfloat16*)s[2];
            int rows=s[6],cols=s[7],ld=s[8],transpose=s[9];
            if(!transpose){
                for(int i=global;i<rows*cols;i+=stride){int r=i/cols,c=i%cols;pack_components(x[r*ld+c],out,i,rows*cols,op==49);}
            }else{
                // Shared transpose makes both global reads and BF16 writes
                // coalesced; padding avoids shared-memory bank conflicts.
                float* tile=workspace[0];int nc=(cols+31)/32;
                for(int task=blockIdx.x;task<((rows+31)/32)*nc;task+=gridDim.x){
                    int r0=(task/nc)*32,c0=(task%nc)*32;
                    for(int i=threadIdx.x;i<1024;i+=blockDim.x){int r=i/32,c=i%32;
                        tile[r*33+c]=(r0+c<rows && c0+r<cols)?x[(c0+r)*ld+r0+c]:0.f;}
                    __syncthreads();
                    for(int i=threadIdx.x;i<1024;i+=blockDim.x){int r=i/32,c=i%32;
                        if(r0+r<rows && c0+c<cols)pack_components(tile[c*33+r],out,(r0+r)*cols+c0+c,rows*cols,op==49);}
                    __syncthreads();
                }
            }
#ifdef CHISEL_CUTLASS
        } else if(op==44) {
            int tasks=((s[6]+127)/128)*((s[7]+63)/64)*s[12];
            while(true){
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                cutlass_bf16_tile(s,first_task,workspace[0]);__syncthreads();
            }
#endif
        }
#ifdef CHISEL_FP32_EMULATION
        else if(op==50) {
            int tasks=((s[6]+127)/128)*((s[7]+63)/64)*s[12];
            while(true) {
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                cutlass_bf16x9_tile(s,first_task,workspace[0]);__syncthreads();
            }
        }
#endif
#ifdef CHISEL_CUTLASS_IEEE
        else if(op==46) {
            int tasks=(s[6]/128)*(s[7]/64)*s[12]*max(1,(int)s[14]);
            while(true) {
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                using R=cutlass::layout::RowMajor;using C=cutlass::layout::ColumnMajor;
                int flags=s[11]&3;
                if(flags==0)cutlass_ieee_tile<R,R>(s,first_task,workspace[0]);
                else if(flags==1)cutlass_ieee_tile<C,R>(s,first_task,workspace[0]);
                else if(flags==2)cutlass_ieee_tile<R,C>(s,first_task,workspace[0]);
                else cutlass_ieee_tile<C,C>(s,first_task,workspace[0]);
                __syncthreads();
            }
        }
#endif
#ifdef CHISEL_CUTLASS_TF32X3
        else if(op==48) {
            int tasks=((s[6]+127)/128)*((s[7]+63)/64)*s[12]*max(1,(int)s[14]);
            while(true) {
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                using R=cutlass::layout::RowMajor;using C=cutlass::layout::ColumnMajor;
                int flags=s[11]&3;
                if(flags==0)cutlass_tf32x3_tile<R,R>(s,first_task,workspace[0]);
                else if(flags==1)cutlass_tf32x3_tile<C,R>(s,first_task,workspace[0]);
                else if(flags==2)cutlass_tf32x3_tile<R,C>(s,first_task,workspace[0]);
                else cutlass_tf32x3_tile<C,C>(s,first_task,workspace[0]);
                __syncthreads();
            }
        }
#endif
        else if(op==45) {
            bool skinny=s[7]<=16 && !s[21];
            int width=skinny?8:64;
            int tasks=((s[6]+63)/64)*((s[7]+width-1)/width)*s[12]*max(1,(int)s[14]);
            while(true) {
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                if(skinny) {
#if defined(CHISEL_IEEE_SKINNY_PIPELINE)
                    using R=cutlass::layout::RowMajor;using C=cutlass::layout::ColumnMajor;
                    int flags=s[11]&3;
                    if((s[11]>>2)>1)ieee_skinny_tile(s,first_task,workspace[0]);
                    else if(flags==0)cutlass_ieee_skinny_tile<R,R>(s,first_task,workspace[0]);
                    else if(flags==1)cutlass_ieee_skinny_tile<C,R>(s,first_task,workspace[0]);
                    else if(flags==2)cutlass_ieee_skinny_tile<R,C>(s,first_task,workspace[0]);
                    else cutlass_ieee_skinny_tile<C,C>(s,first_task,workspace[0]);
#elif defined(CHISEL_IEEE_SKINNY_PARALLEL)
                    ieee_skinny_parallel_tile(s,first_task,workspace[0]);
#else
                    ieee_skinny_tile(s,first_task,workspace[0]);
#endif
                }
                else ieee_cta_tile(s,first_task,workspace[0]);
                __syncthreads();
            }
        } else if(op==GEMM) {
            bool cta_tile=s[6]>=32 && s[7]>=32 && s[8]>=32;
            bool big_tile=s[20] && s[6]>=64 && s[7]>=128 && s[8]>=32;
            int tasks=(big_tile?((s[6]+63)/64)*((s[7]+127)/128):cta_tile?((s[6]+31)/32)*((s[7]+63)/64):((s[6]+15)/16)*((s[7]+15)/16))*s[12]*max(1,(int)s[14]);
            while(true) {
                if(threadIdx.x==0) first_task=atomicAdd(cursor,cta_tile?1:WARPS);
                __syncthreads();
                if(first_task>=tasks) break;
                int task=first_task+threadIdx.x/32;
                if(big_tile)bf16_big_tile(s,first_task,workspace[0]);
                else if(cta_tile) {
                    if(s[20])bf16_cta_tile(s,first_task,workspace[0]);
                    else tf32_cta_tile(s,first_task,workspace[0]);
                }
                else if(task<tasks) {
                    if(s[20]) bf16_tensor_tile(s,task,workspace[threadIdx.x/32]);
                    else tensor_tile(s,task,workspace[threadIdx.x/32],workspace[0],((s[7]+15)/16)==WARPS);
                }
                __syncthreads();
            }
        } else if(op==STREAM_ATTENTION_FORWARD || op==STREAM_ATTENTION_DQ || op==STREAM_ATTENTION_DKDV) {
            int tasks=((s[6]+15)/16)*s[7];
            while(true) {
                if(threadIdx.x==0)first_task=atomicAdd(cursor,1);__syncthreads();
                if(first_task>=tasks)break;
                if(op==STREAM_ATTENTION_FORWARD){if(s[14])attention_forward_tile<true>(s,first_task,workspace[0]);else attention_forward_tile<false>(s,first_task,workspace[0]);}
                else if(op==STREAM_ATTENTION_DQ){if(s[14])attention_dq_tile<true>(s,first_task,workspace[0]);else attention_dq_tile<false>(s,first_task,workspace[0]);}
                else {if(s[14])attention_dkdv_tile<true>(s,first_task,workspace[0]);else attention_dkdv_tile<false>(s,first_task,workspace[0]);}__syncthreads();
            }
        } else if(op==ATTENTION_ROW_DELTA) {
            const float *x=(const float*)s[1],*dy=(const float*)s[2];float* delta=(float*)s[3];
            for(int row=blockIdx.x;row<s[6];row+=gridDim.x) {
                float sum=0.f;for(int d=threadIdx.x;d<128;d+=blockDim.x)sum+=x[row*128+d]*dy[row*128+d];
                sum=block_reduce(sum,workspace[0],false);if(threadIdx.x==0)delta[row]=sum;
            }
        } else if(op==REDUCE) {
            const float* input=(const float*)s[1]; float* output=(float*)s[2];
            int rows=s[6], cols=s[7];
            for(int col=global;col<cols;col+=stride) {
                float sum=0.f;
                for(int row=0;row<rows;row++) sum+=input[row*cols+col];
                output[col]=s[8]?sum*from_bits(s[8]):sum;
            }
        } else if(op==BIAS_PARTIAL) {
            const float* input=(const float*)s[1]; float* output=(float*)s[2];
            int rows=s[6], cols=s[7], chunk=s[8], parts=(rows+chunk-1)/chunk;
            for(int index=global;index<parts*cols;index+=stride) {
                int part=index/cols, col=index%cols;
                float sum=0.f;
                for(int row=part*chunk;row<min(rows,(part+1)*chunk);row++) sum+=input[row*cols+col];
                output[index]=sum;
            }
        } else if(op==ADAM) {
            float *p=(float*)s[1], *g=(float*)s[2], *m=(float*)s[3], *v=(float*)s[4];
            int n=s[6]; float lr=from_bits(s[7]), b1=from_bits(s[8]), b2=from_bits(s[9]), eps=from_bits(s[10]);
            float bc1=1.f-powf(b1,(float)*step), bc2=1.f-powf(b2,(float)*step);
            for(int i=global;i<n;i+=stride) {
                float grad=g[i], moment=m[i]+(1.f-b1)*(grad-m[i]);
                float variance=b2*v[i]+(1.f-b2)*grad*grad;
                m[i]=moment; v[i]=variance;
                p[i]-=lr*(moment/bc1)/(sqrtf(variance/bc2)+eps);
            }
        } else if(op==ADD) {
            const float *a=(const float*)s[1], *b=(const float*)s[2]; float* out=(float*)s[3];
            float alpha=s[7]?from_bits(s[7]):1.f, beta=s[8]?from_bits(s[8]):1.f;
            for(int i=global;i<s[6];i+=stride) out[i]=alpha*a[i]+beta*b[i];
        } else if(op==COPY) {
            const float* source=(const float*)s[1]; float* target=(float*)s[2];
            for(int i=global;i<s[6];i+=stride) target[i]=source[i];
        } else if(op==UNARY) {
            const float *x=(const float*)s[1], *dy=(const float*)s[2];float* out=(float*)s[3];
            int mode=s[7];float scale=from_bits(s[8]);
            for(int i=global;i<s[6];i+=stride) {
                float v=x[i], result;
                if(mode==0) result=bf16(v);
                else if(mode==1) {v=fmaxf(v,0.f);result=bf16(v*v);}
                else if(mode==2) result=v>0.f?bf16(bf16(2.f*v)*dy[i]):0.f;
                else if(mode==3) result=(15.f*v)*rsqrtf(v*v+225.f);
                else if(mode==4) {float inv=rsqrtf(v*v+225.f);result=bf16(dy[i]*(3375.f*inv*inv*inv));}
                else if(mode==6) {v=fmaxf(v,0.f);result=v*v;}
                else if(mode==7) result=v>0.f?(2.f*v)*dy[i]:0.f;
                else if(mode==8) {float inv=rsqrtf(v*v+225.f);result=dy[i]*(3375.f*inv*inv*inv);}
                else result=v*scale;
                out[i]=result;
            }
        } else if(op==RMS_NORM || op==RMS_NORM_BACKWARD) {
            const float *x=(const float*)s[1],*gamma=(const float*)s[2],*dy=(const float*)s[3];
            float *out=(float*)s[4],*stats=(float*)s[5];int rows=s[6],cols=s[7];
            for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                float sum=0.f;
                if(op==RMS_NORM) {
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {float v=x[row*cols+col];sum+=v*v;}
                    float inverse=rsqrtf(block_reduce(sum,workspace[0],false)/cols+from_bits(s[8]));
                    if(threadIdx.x==0) stats[row]=inverse;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float gain=gamma?(s[10]?gamma[col]:bf16(gamma[col])):1.f;
                        float result=x[row*cols+col]*inverse*gain;
                        out[row*cols+col]=s[10]?result:bf16(result);
                    }
                } else {
                    float inverse=stats[row];
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x)
                        sum+=dy[row*cols+col]*(gamma?(s[10]?gamma[col]:bf16(gamma[col])):1.f)*x[row*cols+col];
                    float correction=block_reduce(sum,workspace[0],false)*inverse*inverse/cols;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float gain=gamma?(s[10]?gamma[col]:bf16(gamma[col])):1.f;
                        float result=(dy[row*cols+col]*gain-x[row*cols+col]*correction)*inverse;
                        out[row*cols+col]=s[10]?result:bf16(result);
                        if(s[9]) ((float*)s[9])[row*cols+col]=dy[row*cols+col]*x[row*cols+col]*inverse;
                    }
                }
                __syncthreads();
            }
        } else if(op==ROTARY) {
            const float *x=(const float*)s[1],*frequency=(const float*)s[2];float* out=(float*)s[3];
            int tokens=s[6],heads=s[7],width=s[8],half=width/2,sequence_tokens=s[10]?s[10]:tokens;bool backward=s[9];
            for(int i=global;i<tokens*heads*half;i+=stride) {
                int d=i%half, pair=i/half, token=(pair/heads)%sequence_tokens, offset=pair*width+d;
                float theta=token*frequency[d],co=cosf(theta),si=sinf(theta),a=x[offset],b=x[offset+half];
                if(backward) si=-si;
                float first=a*co+b*si,second=-a*si+b*co;
                out[offset]=s[11]?first:bf16(first);out[offset+half]=s[11]?second:bf16(second);
            }
        } else if(op==ZERO) {
            float* out=(float*)s[1];for(int i=global;i<s[6];i+=stride)out[i]=0.f;
        } else if(op==EMBEDDING_SCATTER) {
            const float* x=(const float*)s[1];const i64* indices=(const i64*)s[2];float* out=(float*)s[3];int tokens=s[6],width=s[8];
            for(int i=global;i<tokens*width;i+=stride)atomicAdd(out+indices[i/width]*width+i%width,x[i]);
        } else if(op==EMBEDDING || op==EMBEDDING_BACKWARD) {
            const float *x=(const float*)s[1];const i64* indices=(const i64*)s[2];float* out=(float*)s[3];
            int tokens=s[6],vocab=s[7],width=s[8];
            if(op==EMBEDDING) {
                int row_stride=s[9]?s[9]:width,col_stride=s[10]?s[10]:1;
                for(int i=global;i<tokens*width;i+=stride) out[i]=x[indices[i/width]*row_stride+(i%width)*col_stride];
            } else {
                for(int i=global;i<vocab*width;i+=stride) {
                    int word=i/width,col=i%width;float sum=0.f;
                    for(int t=0;t<tokens;t++) if(indices[t]==word) sum+=x[t*width+col];
                    out[i]=bf16(sum);
                }
            }
        } else if(op==ATTENTION_DELTA) {
            const float *output=(const float*)s[1],*dy=(const float*)s[2],*probability=(const float*)s[3],*dp=(const float*)s[4];
            float* ds=(float*)s[5];int rows=s[6],keys=s[7],width=s[8];float scale=from_bits(s[9]);
            for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                float sum=0.f;
                for(int d=threadIdx.x;d<width;d+=blockDim.x) sum+=output[row*width+d]*dy[row*width+d];
                float delta=block_reduce(sum,workspace[0],false);
                for(int key=threadIdx.x;key<keys;key+=blockDim.x) ds[row*keys+key]=bf16(probability[row*keys+key]*(dp[row*keys+key]-delta)*scale);
                __syncthreads();
            }
        } else if(op==ROW_DIVIDE) {
            const float *x=(const float*)s[1],*denom=(const float*)s[2];float* out=(float*)s[3];
            for(int i=global;i<s[6]*s[7];i+=stride) out[i]=bf16(x[i]/denom[i/s[7]]);
        } else if(op==FLASH_PROBABILITIES) {
            const float* scores=(const float*)s[1];float *effective=(float*)s[2],*probability=(float*)s[3];
            int rows=s[6],cols=s[7],chunk=s[8];float scale_log2=from_bits(s[9])*1.4426950408889634f;
            for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                int last=row%cols;float maximum=__uint_as_float(0xff800000u);
                for(int col=threadIdx.x;col<=last;col+=blockDim.x) maximum=fmaxf(maximum,scores[row*cols+col]);
                maximum=block_reduce(maximum,workspace[0],true);
                float sum=0.f;
                for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                    float p=col<=last?exp2f(__fmaf_rn(scores[row*cols+col],scale_log2,-maximum*scale_log2)):0.f;
                    probability[row*cols+col]=p;effective[row*cols+col]=0.f;sum+=p;
                }
                sum=block_reduce(sum,workspace[0],false);
                float running_max=__uint_as_float(0xff800000u);
                for(int block=last/chunk;block>=0;block--) {
                    if(s[10] && (block+1)%s[10]==0) running_max=__uint_as_float(0xff800000u);
                    float local=running_max;
                    for(int col=block*chunk+threadIdx.x;col<=min(last,(block+1)*chunk-1);col+=blockDim.x)
                        local=fmaxf(local,scores[row*cols+col]);
                    running_max=block_reduce(local,workspace[0],true);
                    float rescale=exp2f((running_max-maximum)*scale_log2)/sum;
                    for(int col=block*chunk+threadIdx.x;col<=min(last,(block+1)*chunk-1);col+=blockDim.x) {
                        float p=exp2f(__fmaf_rn(scores[row*cols+col],scale_log2,-running_max*scale_log2));
                        effective[row*cols+col]=bf16(p)*rescale;
                    }
                    __syncthreads();
                }
                for(int col=threadIdx.x;col<cols;col+=blockDim.x) probability[row*cols+col]/=sum;
                __syncthreads();
            }
        } else if(op==KL) {
            const float* logits=(const float*)s[1];const double* teacher=(const double*)s[2];
            float* gradient=(float*)s[3];double* losses=(double*)s[4];int rows=s[6],cols=s[7];
            bool softcap=s[8];float gradient_scale=from_bits(s[9]);
            for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                float maximum=__uint_as_float(0xff800000u);
                for(int col=threadIdx.x;col<cols;col+=blockDim.x){float v=logits[row*cols+col];if(softcap)v=(15.f*v)*rsqrtf(v*v+225.f);maximum=fmaxf(maximum,v);}
                maximum=block_reduce(maximum,workspace[0],true);
                double mass=0.,sum=0.;
                for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                    float v=logits[row*cols+col];if(softcap)v=(15.f*v)*rsqrtf(v*v+225.f);
                    sum+=exp((double)v-maximum);mass+=exp(teacher[row*cols+col]);
                }
                sum=block_reduce_double(sum,workspace[0]);mass=block_reduce_double(mass,workspace[0]);
                double logsum=log(sum),loss=0.;
                for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                    int i=row*cols+col;float raw=logits[i],v=softcap?(15.f*raw)*rsqrtf(raw*raw+225.f):raw;
                    double lp=(double)v-maximum-logsum,p=exp(teacher[i]);
                    loss+=p*(teacher[i]-lp);float g=(float)((exp(lp)*mass-p)/rows);
                    if(softcap){float inv=rsqrtf(raw*raw+225.f);g=(g*gradient_scale)*(3375.f*inv*inv*inv);if(!s[10])g=bf16(g);}
                    gradient[i]=g;
                }
                loss=block_reduce_double(loss,workspace[0]);if(threadIdx.x==0) losses[row]=loss/rows;
            }
        } else if(op==DOUBLE_POINTER_SUM) {
            if(global==0){const u64* pointers=(const u64*)s[1];const double* weights=(const double*)s[8];double sum=0.;for(int i=0;i<s[6];i++)sum+=*(const double*)pointers[i]*(weights?weights[i]:1.);*(double*)s[2]=sum*from_bits(s[7]);}
        } else if(op==DOUBLE_REDUCE) {
            const double* x=(const double*)s[1];double* out=(double*)s[2];int rows=s[6],cols=s[7];
            for(int col=global;col<cols;col+=stride) {double sum=0.;for(int row=0;row<rows;row++) sum+=x[row*cols+col];out[col]=sum;}
        } else if(op==PAIR_STATS) {
            const float *a=(const float*)s[1],*b=(const float*)s[2];double* out=(double*)s[3];
            int count=s[6],chunk=s[7],mode=s[8],parts=(count+chunk-1)/chunk;
            for(int part=blockIdx.x;part<parts;part+=gridDim.x) {
                double first=0.,second=0.;
                for(int i=part*chunk+threadIdx.x;i<min(count,(part+1)*chunk);i+=blockDim.x) {
                    double x=a[i];
                    if(mode==0) first+=x*x;
                    else if(mode==1) first+=x*(double)b[i];
                    else {double base=b[i],delta=x-base;first+=delta*(x+base);second+=delta*delta;}
                }
                first=block_reduce_double(first,workspace[0]);second=block_reduce_double(second,workspace[0]);
                if(threadIdx.x==0) {out[2*part]=first;out[2*part+1]=second;}
            }
        } else if(op==RADIUS || op==RADIAL_COTANGENT) {
            const u64* pointers=(const u64*)s[1];const float* info=(const float*)s[2];float* out=(float*)s[3];
            if(global==0) {
                double sum=0.;for(int i=0;i<s[6];i++) sum+=*(const double*)pointers[i];
                float radius=from_bits(s[7]);
                if(op==RADIUS) {float norm=sqrtf(fmaxf((float)sum,1e-24f));out[0]=norm;out[1]=fminf(1.f,radius/(norm+1e-12f));}
                else {float norm=info[0];out[0]=info[1]<1.f?(float)sum*(-radius/((norm+1e-12f)*(norm+1e-12f)))/norm:0.f;}
            }
        } else if(op==EDIT) {
            const float *base=(const float*)s[1],*delta=(const float*)s[2],*info=(const float*)s[3];float* out=(float*)s[4];
            for(int i=global;i<s[6];i+=stride) {float v=base[i]+info[1]*delta[i];out[i]=s[7]?bf16(v):v;}
        } else if(op==GROUP_OBJECTIVE) {
            const u64* pointers=(const u64*)s[1];const i64* groups=(const i64*)s[2];const double* refs=(const double*)s[3];double* out=(double*)s[4];
            for(int group=global;group<s[7];group+=stride) {
                double delta=0.,edit=0.;for(int i=0;i<s[6];i++) if(groups[i]==group) {const double* values=(const double*)pointers[i];delta+=values[0];edit+=values[1];}
                double old=refs[group],norm=sqrt(fmax(old+delta,0.)),oldnorm=sqrt(old);
                out[group*3]=old>0.?delta/(oldnorm*(oldnorm+norm)):0.;out[group*3+1]=edit;
                out[group*3+2]=(old>0. && norm>0.)?100./(s[7]*oldnorm*norm):0.;
            }
        } else if(op==OBJECTIVE_COMBINE) {
            const double *kl=(const double*)s[1],*groups=(const double*)s[2];double *out=(double*)s[3],*coefficients=(double*)s[4];
            if(global==0) {
                double budget=*(const double*)s[5],radius=*(const double*)s[8],norm=0.,trust=0.;
                const double* controls=(const double*)s[9];double trust_weight=controls?controls[0]:.01;
                for(int g=0;g<s[6];g++) {norm+=groups[g*3]*100./s[6];trust+=groups[g*3+1]*trust_weight/(radius*radius);}
                double excess=fmax(*kl/budget-1.,0.);out[0]=*kl;out[1]=excess*excess;out[2]=norm;out[3]=trust;out[4]=out[1]+norm+trust;
                coefficients[0]=2.*excess/budget;coefficients[1]=2.*trust_weight/(radius*radius);
                if(controls && controls[3]==1.) {
                    double violation=fmax(*kl,0.)/budget-1.,positive=fmax(violation,0.);
                    out[1]=controls[1]*violation+.5*controls[2]*positive*positive;
                    out[4]=out[1]+norm+trust;
                    coefficients[0]=*kl>=0.?(controls[1]+controls[2]*positive)/budget:0.;
                } else if(controls && controls[3]==2.) {
                    out[1]=*kl;out[2]=0.;out[3]=0.;out[4]=*kl;
                    coefficients[0]=1.;coefficients[1]=0.;
                }
            }
        } else if(op==PARAMETER_COTANGENT) {
            const float *base=(const float*)s[1],*value=(const float*)s[2],*klgrad=(const float*)s[3];
            const double *groups=(const double*)s[4],*coefficients=(const double*)s[5];float* out=(float*)s[9];
            double weight=s[10]?from_bits(s[10]):1.,norm=groups[s[7]*3+2]*weight,trust=coefficients[1]*weight,kl=coefficients[0]*weight;
            if(s[11])norm=0.;
            for(int i=global;i<s[6];i+=stride) {
                float regularizer=(float)(norm*value[i]+trust*((double)value[i]-base[i]));
                float scaled=klgrad[i]*(float)kl;
                out[i]=s[8]?bf16(bf16(regularizer)+bf16(scaled)):regularizer+scaled;
            }
        } else if(op==47) {
            const float *klgrad=(const float*)s[1],*normgrad=(const float*)s[2];float* out=(float*)s[3];
            const double *groups=(const double*)s[4],*coefficients=(const double*)s[5];
            double weight=from_bits(s[9]),norm=s[8]?0.:groups[s[7]*3+2]*weight,kl=coefficients[0]*weight;
            for(int i=global;i<s[6];i+=stride)out[i]=(float)(norm*normgrad[i])+klgrad[i]*(float)kl;
        } else if(op==FACTOR_COTANGENT) {
            const float *x=(const float*)s[1],*dy=(const float*)s[2],*info=(const float*)s[3],*radial=(const float*)s[4];float* out=(float*)s[5];
            float scale=from_bits(s[7]);
            for(int i=global;i<s[6];i+=stride) out[i]=(info[1]*dy[i]+radial[0]*x[i])*scale;
        } else if(op==CLIP_GRADIENT) {
            const u64* pointers=(const u64*)s[1];float* gradient=(float*)s[2];
            double sum=0.;for(int i=0;i<s[7];i++) sum+=*(const double*)pointers[i];
            float scale=fminf(1.f,from_bits(s[8])/((float)sqrt(sum)+1e-6f));
            for(int i=global;i<s[6];i+=stride) gradient[i]*=scale;
        } else if(op==BROADCAST) {
            const float* source=(const float*)s[1]; float* target=(float*)s[2];
            int rows=s[6], cols=s[7]; float scale=from_bits(s[8]);
            for(int i=global;i<rows*cols;i+=stride) target[i]=source[i%cols]*scale;
        } else if(op==CONTEXT_CONCAT || op==CONTEXT_SPLIT) {
            int rows=s[6], local=s[7], context=s[8], width=local+context;
            const float* a=(const float*)s[1];
            float *b=(float*)s[2], *c=(float*)s[3];
            for(int i=global;i<rows*width;i+=stride) {
                int row=i/width,col=i%width;
                if(op==CONTEXT_CONCAT) c[i]=col<local?a[row*local+col]:b[col-local];
                else if(col<local) b[row*local+col]=a[i];
                else c[row*context+col-local]=a[i];
            }
        } else if(op==TRAJECTORY_SCALE) {
            const float* input=(const float*)s[1];float *scales=(float*)s[2],*logs=(float*)s[3];
            int count=s[6]/2,rank=s[7];
            if(blockIdx.x<2) {
                int bank=blockIdx.x;float sum=0.f;
                for(int i=threadIdx.x;i<count;i+=blockDim.x) {
                    int index=(i/rank)*2*rank+bank*rank+i%rank;
                    float value=input[index];sum+=value*value;
                }
                sum=block_reduce(sum,workspace[0],false);
                if(threadIdx.x==0) {float scale=fmaxf(sqrtf(sum/count),1e-12f);scales[bank]=scale;logs[bank]=logf(scale);}
            }
        } else if(op==TRAJECTORY_NORMALIZE) {
            const float *input=(const float*)s[1], *scales=(const float*)s[2];float* output=(float*)s[3];
            int count=s[6],rank=s[7];
            for(int i=global;i<count;i+=stride) output[i]=input[i]/scales[(i/rank)%2];
        } else if(op==HEAD_LAYOUT) {
            const float* input=(const float*)s[1]; float* output=(float*)s[2];
            int heads=s[6], tokens=s[7], width=s[8], mode=s[9];
            int parts=mode<2?3:1,batches=s[10]?s[10]:1,elements=batches*parts*heads*tokens*width;
            for(int i=global;i<elements;i+=stride) {
                int d=i%width, index=i/width, source;
                if(mode==0 || mode==3) {
                    int token=index%tokens, head=(index/tokens)%heads,part=(index/(tokens*heads))%parts,batch=index/(tokens*heads*parts);
                    source=(((batch*tokens+token)*parts+part)*heads+head)*width+d;
                } else {
                    int head=index%heads,part=(index/heads)%parts,token=(index/(heads*parts))%tokens,batch=index/(heads*parts*tokens);
                    source=(((batch*parts+part)*heads+head)*tokens+token)*width+d;
                }
                output[i]=input[source];
            }
        } else if(op==LAYER_NORM || op==LAYER_NORM_BACKWARD) {
            const float* x=(const float*)s[1];
            int rows=s[6], cols=s[7];
            float* stats=(float*)s[5];
            if(op==LAYER_NORM) {
                const float *gamma=(const float*)s[2], *beta=(const float*)s[3]; float* out=(float*)s[4];
                for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                    float sum=0.f;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) sum+=x[row*cols+col];
                    float mean=block_reduce(sum,workspace[0],false)/cols;
                    float squared=0.f;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float centered=x[row*cols+col]-mean; squared+=centered*centered;
                    }
                    float variance=block_reduce(squared,workspace[0],false)/cols;
                    float inverse=rsqrtf(variance+from_bits(s[8]));
                    if(threadIdx.x==0) {stats[2*row]=mean;stats[2*row+1]=inverse;}
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x)
                        out[row*cols+col]=(x[row*cols+col]-mean)*inverse*gamma[col]+beta[col];
                    __syncthreads();
                }
            } else {
                const float *dy=(const float*)s[2], *gamma=(const float*)s[3];
                float *dx=(float*)s[4], *dg=(float*)s[10];
                for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                    float mean=stats[2*row], inverse=stats[2*row+1], sum=0.f, correlated=0.f;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float normalized=(x[row*cols+col]-mean)*inverse;
                        float d=dy[row*cols+col]*gamma[col];sum+=d;correlated+=d*normalized;
                    }
                    float average=block_reduce(sum,workspace[0],false)/cols;
                    float correlation=block_reduce(correlated,workspace[0],false)/cols;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float normalized=(x[row*cols+col]-mean)*inverse;
                        dx[row*cols+col]=(dy[row*cols+col]*gamma[col]-average-correlation*normalized)*inverse;
                        dg[row*cols+col]=dy[row*cols+col]*normalized;
                    }
                    __syncthreads();
                }
            }
        } else if(op==SOFTMAX || op==SOFTMAX_BACKWARD) {
            const float* input=(const float*)s[1];
            const float* cotangent=(const float*)s[2];
            float* output=(float*)s[3];
            int rows=s[6], cols=s[7], causal=s[8], query_length=s[9];
            for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
                if(op==SOFTMAX) {
                    float maximum=__uint_as_float(0xff800000u);
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x)
                        if(!causal || col<=row%query_length) maximum=fmaxf(maximum,input[row*cols+col]);
                    maximum=block_reduce(maximum,workspace[0],true);
                    float sum=0.f;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) {
                        float value=(!causal || col<=row%query_length)?expf(input[row*cols+col]-maximum):0.f;
                        output[row*cols+col]=value; sum+=value;
                        if(s[10]) ((float*)s[10])[row*cols+col]=bf16(value);
                    }
                    sum=block_reduce(sum,workspace[0],false);
                    if(s[11] && threadIdx.x==0) ((float*)s[11])[row]=sum;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) output[row*cols+col]/=sum;
                } else {
                    float dot=0.f;
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x) dot+=input[row*cols+col]*cotangent[row*cols+col];
                    dot=block_reduce(dot,workspace[0],false);
                    for(int col=threadIdx.x;col<cols;col+=blockDim.x)
                        output[row*cols+col]=input[row*cols+col]*(cotangent[row*cols+col]-dot);
                }
                __syncthreads();
            }
        }
        // All output tiles/reductions are globally visible before any consumer
        // stage. Cooperative launch is occupancy bounded, so this cannot wait
        // on an unscheduled CTA as an ordinary oversubscribed grid would.
        grid.sync();
        if(global==0) { *cursor=0; clocks[stage+1]=clock64(); }
        grid.sync();
    }
}
