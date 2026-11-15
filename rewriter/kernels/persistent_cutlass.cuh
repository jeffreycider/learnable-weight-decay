// Optional CUTLASS 3.9.2 threadblock primitive, invoked as a device function
// inside persistent_training. No CUTLASS host launch or child kernel is used.
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

using ChiselMmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<128,64,32>, cutlass::gemm::GemmShape<64,32,32>,
    cutlass::gemm::GemmShape<16,8,16>, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, 3>;
using ChiselIteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<128,32>, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    1, ChiselMmaCore::IteratorThreadMapA,
    cutlass::Array<cutlass::bfloat16_t,ChiselMmaCore::IteratorThreadMapA::kElementsPerAccess>>;
using ChiselIteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<32,64>, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    0, ChiselMmaCore::IteratorThreadMapB,
    cutlass::Array<cutlass::bfloat16_t,ChiselMmaCore::IteratorThreadMapB::kElementsPerAccess>>;
using ChiselMma = cutlass::gemm::threadblock::MmaMultistage<
    ChiselMmaCore::Shape, ChiselIteratorA, ChiselMmaCore::SmemIteratorA,
    ChiselMmaCore::kCacheOpA, ChiselIteratorB, ChiselMmaCore::SmemIteratorB,
    ChiselMmaCore::kCacheOpB, float, cutlass::layout::RowMajor,
    ChiselMmaCore::MmaPolicy, 3>;
static_assert(sizeof(ChiselMma::SharedStorage)<=4*2816*sizeof(float),"CUTLASS tile exceeds persistent workspace");

__device__ void cutlass_bf16_tile(const unsigned long long* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],tiles_m=(M+127)/128,tiles_n=(N+63)/64;
    int part=task/(tiles_m*tiles_n),tile=task%(tiles_m*tiles_n);
    int tm=tile/tiles_n,tn=tile%tiles_n,k0=part*(int)s[13];
    int length=min((int)s[13],K-k0),tid=threadIdx.x,warp=tid/32,lane=tid%32;
    auto* a=(cutlass::bfloat16_t*)s[1];auto* b=(cutlass::bfloat16_t*)s[2];
    ChiselIteratorA ia(ChiselIteratorA::Params(cutlass::layout::RowMajor(K)),a+k0,
        {M,length},tid,{tm*128,0});
    ChiselIteratorB ib(ChiselIteratorB::Params(cutlass::layout::ColumnMajor(K)),b+k0,
        {length,N},tid,{0,tn*64});
    ChiselMma mma(*reinterpret_cast<ChiselMma::SharedStorage*>(sh),tid,warp,lane);
    ChiselMma::FragmentC accum;accum.clear();
    mma((length+31)/32,accum,ia,ib,accum);
    float* out=(float*)s[3]+part*M*N;
    // Reuse the retired input pipeline storage for a predicated epilogue.
    // Direct IteratorC global stores would overrun the final partial tile.
    __syncthreads();
    ChiselMma::Operator::IteratorC ic({sh,64},lane);
    ic.add_tile_offset({warp%2,warp/2});ic.store(accum);
    __syncthreads();
    float factor=__uint_as_float((unsigned)s[19]);const float* bias=(const float*)s[4];
    for(int i=tid;i<128*64;i+=blockDim.x){
        int r=tm*128+i/64,c=tn*64+i%64;
        if(r<M && c<N){
            float value=sh[i]*factor+(bias?bias[c]:0.f);
            out[r*N+c]=s[22]?__bfloat162float(__float2bfloat16_rn(value)):value;
        }
    }
    __syncthreads();
}
