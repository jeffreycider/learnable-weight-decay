// Experimental FP32-input/output emulation using three BF16 residuals per
// operand and ALL nine products. This is not a single reduced-precision cast.
// Each product is reduced independently before adding its FP32 result, avoiding
// loss of low corrections inside a subsequently cancelling high-product dot.
__device__ void cutlass_bf16x9_tile(const unsigned long long* s,int task,float* sh) {
    int M=s[6],N=s[7],K=s[8],tm=(M+127)/128,tn=(N+63)/64;
    int part=task/(tm*tn),tile=task%(tm*tn),row=tile/tn,col=tile%tn;
    int k0=part*(int)s[13],length=min((int)s[13],K-k0),tid=threadIdx.x,warp=tid/32,lane=tid%32;
    auto* a=(cutlass::bfloat16_t*)s[1];auto* b=(cutlass::bfloat16_t*)s[2];
    ChiselMma::FragmentC sum;sum.clear();
    constexpr int ac[9]={0,0,1,1,0,2,1,2,2};
    constexpr int bc[9]={0,1,0,1,2,0,2,1,2};
    #pragma unroll 1
    for(int product=0;product<9;product++) {
        ChiselIteratorA ia(ChiselIteratorA::Params(cutlass::layout::RowMajor(K)),a+ac[product]*M*K+k0,
            {M,length},tid,{row*128,0});
        ChiselIteratorB ib(ChiselIteratorB::Params(cutlass::layout::ColumnMajor(K)),b+bc[product]*K*N+k0,
            {length,N},tid,{0,col*64});
        ChiselMma mma(*reinterpret_cast<ChiselMma::SharedStorage*>(sh),tid,warp,lane);
        ChiselMma::FragmentC accum;accum.clear();
        mma((length+31)/32,accum,ia,ib,accum);
        __syncthreads();
        #pragma unroll
        for(int i=0;i<ChiselMma::FragmentC::kElements;i++)sum[i]=__fadd_rn(sum[i],accum[i]);
    }
    ChiselMma::Operator::IteratorC ic({sh,64},lane);
    ic.add_tile_offset({warp%2,warp/2});ic.store(sum);
    __syncthreads();
    float* out=(float*)s[3]+part*M*N;const float* bias=(const float*)s[4];
    float factor=__uint_as_float((unsigned)s[19]);
    for(int i=tid;i<8192;i+=blockDim.x) {
        int r=row*128+i/64,c=col*64+i%64;
        if(r<M && c<N)out[r*N+c]=sh[i]*factor+(bias?bias[c]:0.f);
    }
    __syncthreads();
}
