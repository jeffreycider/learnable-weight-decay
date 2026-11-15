// CUTLASS3.9.2 SIMT threadblock pipeline, called inside the cooperative kernel.
// Explicit IEEE FP32 FMA preserves operand bits even with global --fmad=false.
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma.h"

namespace cutlass { namespace arch {
struct ChiselIEEEMultiplyAdd {};
template<class LA,class LB,class LC>
struct Mma<gemm::GemmShape<1,1,1>,1,float,LA,float,LB,float,LC,ChiselIEEEMultiplyAdd> {
    using Shape=gemm::GemmShape<1,1,1>;
    using Operator=ChiselIEEEMultiplyAdd;
    using ElementC=float;
    CUTLASS_DEVICE void operator()(Array<float,1>& d,Array<float,1> const& a,
                                  Array<float,1> const& b,Array<float,1> const& c) {
        d[0]=__fmaf_rn(a[0],b[0],c[0]);
    }
};
}}
namespace cutlass { namespace gemm { namespace thread {
template<class Shape,class LA,class LB,class LC>
struct Mma<Shape,float,LA,float,LB,float,LC,arch::ChiselIEEEMultiplyAdd,bool>
    : MmaGeneric<Shape,float,LA,float,LB,float,LC,arch::ChiselIEEEMultiplyAdd> {
    using ArchMmaOperator=typename MmaGeneric<Shape,float,LA,float,LB,float,LC,arch::ChiselIEEEMultiplyAdd>::MmaOp;
};
// CUTLASS MmaSimt hardcodes OpMultiplyAdd internally, regardless of the
// DefaultMmaCore operator tag. Route that actual thread specialization through
// the explicit IEEE FMA too; --fmad=false must not split this contraction.
template<class Shape,class LA,class LB,class LC>
struct Mma<Shape,float,LA,float,LB,float,LC,arch::OpMultiplyAdd,bool>
    : MmaGeneric<Shape,float,LA,float,LB,float,LC,arch::ChiselIEEEMultiplyAdd> {
    using ArchMmaOperator=typename MmaGeneric<Shape,float,LA,float,LB,float,LC,arch::ChiselIEEEMultiplyAdd>::MmaOp;
};
}}}
#ifdef CHISEL_IEEE_DEEP
#include "persistent_ieee_deep.cuh"
#endif

template<class LA,class LB>
__device__ void cutlass_ieee_tile(const unsigned long long* s,int task,float* sh) {
    using Default=cutlass::gemm::threadblock::DefaultMma<
        float,LA,1,float,LB,1,float,cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,64,8>,cutlass::gemm::GemmShape<32,64,8>,
        cutlass::gemm::GemmShape<1,1,1>,2,cutlass::arch::ChiselIEEEMultiplyAdd>;
#ifdef CHISEL_IEEE_DEEP
    using Mma=typename ChiselIeeeDeepPipeline<LA,LB>::Mma;
    constexpr int TileK=16;
#else
    using Mma=typename Default::ThreadblockMma;
    constexpr int TileK=8;
#endif
    using IA=typename Mma::IteratorA;using IB=typename Mma::IteratorB;
    static_assert(sizeof(typename Mma::SharedStorage)<=4*2816*sizeof(float),"IEEE tile exceeds workspace");
    int M=s[6],N=s[7],K=s[8],tm=M/128,tn=N/64,parts=s[12];
    int batch=task/(tm*tn*parts);task%=tm*tn*parts;
    int split=task/(tm*tn),tile=task%(tm*tn),row=tile/tn,col=tile%tn;
    int k0=split*(int)s[13],length=min((int)s[13],K-k0),tid=threadIdx.x,warp=tid/32,lane=tid%32;
    if(s[21]==1 && col*64>=row*128+128)length=0;
    if(s[21]==2)length=max(0,min(length,row*128+128-k0));
    if(s[21]==3){int end=k0+length;k0=max(k0,row*128);length=max(0,end-k0);}
    int lda=s[9],ldb=s[10],flags=s[11];
    float* a=(float*)s[1]+batch*s[15]+((flags&1)?k0*lda:k0);
    float* b=(float*)s[2]+batch*s[16]+((flags&2)?k0:k0*ldb);
    IA ia(typename IA::Params(LA(lda)),a,{M,length},tid,{row*128,0});
    IB ib(typename IB::Params(LB(ldb)),b,{length,N},tid,{0,col*64});
    Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(sh),tid,warp,lane);
    typename Mma::FragmentC accum;accum.clear();
    if(length)mma((length+TileK-1)/TileK,accum,ia,ib,accum);
    float* out=(float*)s[3]+batch*s[17]+split*M*N;
    typename Mma::Operator::IteratorC ic({out,N},lane);
    ic.add_tile_offset({row*4+warp,col});ic.store(accum);
    __syncthreads();
    float factor=__uint_as_float((unsigned)s[19]);const float* bias=(const float*)s[4];
    if(factor!=1.f || bias)for(int i=tid;i<128*64;i+=blockDim.x) {
        int r=row*128+i/64,c=col*64+i%64;
        out[r*N+c]=out[r*N+c]*factor+(bias?bias[c]:0.f);
    }
    __syncthreads();
}

#ifdef CHISEL_IEEE_SKINNY_PIPELINE
#include "persistent_ieee_skinny.cuh"
#endif
