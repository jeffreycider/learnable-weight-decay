// Optional deeper IEEE SIMT pipeline. CUTLASS's default K16 transposed tile
// chooses padding2, incompatible with its four-float lane loads. Explicit
// padding4 preserves that vector layout for every physical operand layout.
template<class LA,class LB>
struct ChiselIeeeDeepPipeline {
    using R=cutlass::layout::RowMajor;using C=cutlass::layout::ColumnMajor;
    using Shape=cutlass::gemm::GemmShape<128,64,16>;
    using Warp=cutlass::gemm::GemmShape<32,64,16>;
    static constexpr bool ar=cutlass::platform::is_same<LA,R>::value;
    static constexpr bool bc=cutlass::platform::is_same<LB,C>::value;
    using MapA=cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<ar?16:128,ar?128:16>,128,1>;
    using MapB=cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<bc?16:64,bc?64:16>,128,1>;
    using SmemMapA=typename cutlass::platform::conditional<ar,
        cutlass::transform::TransposePitchLinearThreadMapSimt<MapA>,MapA>::type;
    using SmemMapB=typename cutlass::platform::conditional<bc,
        cutlass::transform::TransposePitchLinearThreadMapSimt<MapB>,MapB>::type;
    using SA=cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<128,16>,float,C,1,SmemMapA>;
    using SB=cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<16,64>,float,R,0,SmemMapB>;
    using IA=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<128,16>,float,LA,1,MapA,1>;
    using IB=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<16,64>,float,LB,0,MapB,1>;
    using WarpPolicy=cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4,8>,
        cutlass::layout::RowMajorInterleaved<2>,cutlass::gemm::GemmShape<4,4,1>>;
    using WarpMma=cutlass::gemm::warp::MmaSimt<Warp,float,C,float,R,float,R,WarpPolicy>;
    using Policy=cutlass::gemm::threadblock::MmaPolicy<WarpMma,
        cutlass::MatrixShape<4,0>,cutlass::MatrixShape<0,4>,1>;
    using Mma=cutlass::gemm::threadblock::MmaPipelined<Shape,IA,SA,IB,SB,float,R,Policy>;
};
