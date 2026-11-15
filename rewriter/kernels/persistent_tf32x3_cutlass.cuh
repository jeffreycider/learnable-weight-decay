// Three compensated TF32 tensor-core products; only explicitly approximate
// contractions use this path. Strict IEEE forward never selects it; the
// separately labeled experimental forward-emulation opt-in may do so.
#include "cutlass/gemm/threadblock/default_mma.h"
#ifndef CHISEL_TF32_TILE_K
#define CHISEL_TF32_TILE_K 16
#endif

template<class LA,class LB>
__device__ void cutlass_tf32x3_tile(const unsigned long long* s,int task,float* sh) {
    using Default=cutlass::gemm::threadblock::DefaultMma<
        float,LA,4,float,LB,4,float,cutlass::layout::RowMajor,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,64,CHISEL_TF32_TILE_K>,cutlass::gemm::GemmShape<64,32,CHISEL_TF32_TILE_K>,
        cutlass::gemm::GemmShape<16,8,8>,3,cutlass::arch::OpMultiplyAddFastF32>;
    using Mma=typename Default::ThreadblockMma;
    using IA=typename Mma::IteratorA;using IB=typename Mma::IteratorB;
#ifdef CHISEL_DYNAMIC_SHARED_BYTES
    static_assert(sizeof(typename Mma::SharedStorage)<=CHISEL_DYNAMIC_SHARED_BYTES,"TF32x3 tile exceeds dynamic workspace");
#else
    static_assert(sizeof(typename Mma::SharedStorage)<=4*2816*sizeof(float),"TF32x3 tile exceeds workspace");
#endif
    int M=s[6],N=s[7],K=s[8],tm=(M+127)/128,tn=(N+63)/64,parts=s[12];
    int batch=task/(tm*tn*parts);task%=tm*tn*parts;
    int split=task/(tm*tn),tile=task%(tm*tn),row=tile/tn,col=tile%tn;
    int k0=split*(int)s[13],length=min((int)s[13],K-k0),tid=threadIdx.x,warp=tid/32,lane=tid%32;
    int lda=s[9],ldb=s[10],flags=s[11];
    float* a=(float*)s[1]+batch*s[15]+((flags&1)?k0*lda:k0);
    float* b=(float*)s[2]+batch*s[16]+((flags&2)?k0:k0*ldb);
    IA ia(typename IA::Params(LA(lda)),a,{M,length},tid,{row*128,0});
    IB ib(typename IB::Params(LB(ldb)),b,{length,N},tid,{0,col*64});
    Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(sh),tid,warp,lane);
    typename Mma::FragmentC accum;accum.clear();
    mma((length+CHISEL_TF32_TILE_K-1)/CHISEL_TF32_TILE_K,accum,ia,ib,accum);
    __syncthreads();
    typename Mma::Operator::IteratorC ic({sh,64},lane);
    ic.add_tile_offset({warp%2,warp/2});ic.store(accum);
    __syncthreads();
    float* out=(float*)s[3]+batch*s[17]+split*M*N;
    float factor=__uint_as_float((unsigned)s[19]);const float* bias=(const float*)s[4];
    for(int i=tid;i<128*64;i+=blockDim.x) {
        int r=row*128+i/64,c=col*64+i%64;
        if(r<M && c<N)out[r*N+c]=sh[i]*factor+(bias?bias[c]:0.f);
    }
    __syncthreads();
}
