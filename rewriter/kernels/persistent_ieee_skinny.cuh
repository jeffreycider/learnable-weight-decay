// Optional double-buffered SIMT pipeline for strict rank-eight contractions.
// Called from the existing cooperative kernel; no extra launch or reduced cast.
template<class LA,class LB>
__device__ void cutlass_ieee_skinny_tile(const unsigned long long* s,int task,float* sh) {
    using Default=cutlass::gemm::threadblock::DefaultMma<
        float,LA,1,float,LB,1,float,cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64,8,16>,cutlass::gemm::GemmShape<16,8,16>,
        cutlass::gemm::GemmShape<1,1,1>,2,cutlass::arch::ChiselIEEEMultiplyAdd>;
    using Mma=typename Default::ThreadblockMma;using IA=typename Mma::IteratorA;using IB=typename Mma::IteratorB;
    static_assert(sizeof(typename Mma::SharedStorage)<=4*2816*sizeof(float),"skinny pipeline exceeds workspace");
    int M=s[6],N=s[7],K=s[8],tm=(M+63)/64,tn=(N+7)/8,parts=s[12];
    int batch=task/(tm*tn*parts);task%=tm*tn*parts;
    int split=task/(tm*tn),tile=task%(tm*tn),row=tile/tn,col=tile%tn;
    int k0=split*(int)s[13],length=min((int)s[13],K-k0),tid=threadIdx.x,warp=tid/32,lane=tid%32;
    int lda=s[9],ldb=s[10],flags=s[11];
    float* a=(float*)s[1]+batch*s[15]+((flags&1)?k0*lda:k0);
    float* b=(float*)s[2]+batch*s[16]+((flags&2)?k0:k0*ldb);
    IA ia(typename IA::Params(LA(lda)),a,{M,length},tid,{row*64,0});
    IB ib(typename IB::Params(LB(ldb)),b,{length,N},tid,{0,col*8});
    Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(sh),tid,warp,lane);
    typename Mma::FragmentC accum;accum.clear();mma((length+15)/16,accum,ia,ib,accum);
    __syncthreads();
    typename Mma::Operator::IteratorC ic({sh,8},lane);ic.add_tile_offset({warp,0});ic.store(accum);
    __syncthreads();
    float* out=(float*)s[3]+batch*s[17]+split*M*N;const float* bias=(const float*)s[4];
    float factor=__uint_as_float((unsigned)s[19]);
    for(int i=tid;i<512;i+=blockDim.x){int r=row*64+i/8,c=col*8+i%8;
        if(r<M && c<N)out[r*N+c]=sh[i]*factor+(bias?bias[c]:0.f);}
    __syncthreads();
}
