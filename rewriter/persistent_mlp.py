"""Actual single cooperative CUDA launch for a complete two-layer MLP update.

This wrapper covers one shared MLP with externally supplied output cotangent,
its forward/backward, and Adam. The full offline program is constructed by
rewriter.persistent_offline using the same cooperative kernel loader.
No torch.compile, CUDA graphs, or device library launches are used.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import struct

import torch


def _check(result, message):
    status, *values = result
    if int(status):
        raise RuntimeError(f'{message}: CUDA status {status}')
    return values[0] if len(values)==1 else values


class CooperativeTrainingKernel:
    """NVRTC compile/load and one occupancy-safe cooperative launch."""
    def __init__(self, *, blocks=None):
        from cuda.bindings import driver, nvrtc
        import nvidia.cuda_runtime
        import triton
        self.driver = driver
        torch.cuda.init()
        self.device=torch.cuda.current_device()
        _check(driver.cuInit(0),'initialize CUDA driver')
        device=_check(driver.cuDeviceGet(self.device),'select CUDA device')
        self.context=_check(driver.cuDevicePrimaryCtxRetain(device),'retain PyTorch primary context')
        _check(driver.cuCtxSetCurrent(self.context),'set primary context for module compilation')
        source=(Path(__file__).parent/'kernels'/'persistent_training.cu').read_text()
        major,minor=torch.cuda.get_device_capability()
        include=Path(nvidia.cuda_runtime.__file__).parent/'include'
        include_dirs=[include,Path(triton.__file__).parent/'backends'/'nvidia'/'include']
        include_dirs.append(Path(__file__).parent/'kernels')
        if os.environ.get('CHISEL_CUTLASS_INCLUDE'):
            include_dirs.append(Path(os.environ['CHISEL_CUTLASS_INCLUDE']))
        if os.environ.get('CHISEL_CUTLASS_IEEE') and not os.environ.get('CHISEL_CUTLASS_INCLUDE'):
            raise ValueError('CUTLASS IEEE pipeline requires pinned CUTLASS includes')
        if os.environ.get('CHISEL_CUDA_CCCL_INCLUDE'):
            include_dirs.append(Path(os.environ['CHISEL_CUDA_CCCL_INCLUDE']))
        elif importlib.util.find_spec('nvidia.cuda_cccl') is not None:
            import nvidia.cuda_cccl
            include_dirs.append(Path(nvidia.cuda_cccl.__file__).parent/'include')
        options=[f'--gpu-architecture=sm_{major}{minor}'.encode(),b'--std=c++17',
                 *[f'--include-path={path}'.encode() for path in include_dirs],
                 b'--fmad=false',b'--device-as-default-execution-space']
        if os.environ.get('CHISEL_PERSISTENT_MAX_REGISTERS'):
            options.append(('--maxrregcount='+os.environ['CHISEL_PERSISTENT_MAX_REGISTERS']).encode())
        if os.environ.get('CHISEL_CUTLASS_IEEE'):
            options.append(b'-DCHISEL_CUTLASS_IEEE=1')
        skinny_k=int(os.environ.get('CHISEL_IEEE_SKINNY_K','32'))
        if skinny_k not in (32,64,128):raise ValueError('skinny K tile must be32,64 or128')
        options.append(('-DCHISEL_IEEE_SKINNY_K='+str(skinny_k)).encode())
        if os.environ.get('CHISEL_IEEE_SKINNY_PIPELINE'):
            if not os.environ.get('CHISEL_CUTLASS_IEEE'):raise ValueError('skinny pipeline requires CUTLASS IEEE')
            options.append(b'-DCHISEL_IEEE_SKINNY_PIPELINE=1')
        if os.environ.get('CHISEL_IEEE_SKINNY_PARALLEL'):
            options.append(b'-DCHISEL_IEEE_SKINNY_PARALLEL=1')
        if os.environ.get('CHISEL_IEEE_DEEP'):
            options.append(b'-DCHISEL_IEEE_DEEP=1')
        if os.environ.get('CHISEL_CUTLASS_TF32X3'):
            if not os.environ.get('CHISEL_CUTLASS_INCLUDE'):
                raise ValueError('CUTLASS TF32x3 pipeline requires pinned CUTLASS includes')
            options.append(b'-DCHISEL_CUTLASS_TF32X3=1')
        tile_k=int(os.environ.get('CHISEL_TF32_TILE_K','16'))
        if tile_k not in (16,32):raise ValueError('compensated tile K must be16 or32')
        emulation=os.environ.get('CHISEL_FP32_EMULATION')
        if emulation not in (None,'bf16x9','tf32x3'):raise ValueError('unknown FP32 emulation backend')
        if emulation and not os.environ.get('CHISEL_CUTLASS_INCLUDE'):raise ValueError('FP32 emulation requires pinned CUTLASS includes')
        if emulation=='tf32x3' and not os.environ.get('CHISEL_CUTLASS_TF32X3'):raise ValueError('TF32x3 forward emulation requires its pinned CUTLASS pipeline')
        if emulation=='bf16x9':options.append(b'-DCHISEL_FP32_EMULATION=1')
        self.dynamic_shared_bytes=73728 if tile_k==32 else 0
        if tile_k==32:options.append(b'-DCHISEL_TF32_TILE_K=32')
        if self.dynamic_shared_bytes:options.append(b'-DCHISEL_DYNAMIC_SHARED_BYTES=73728')
        if os.environ.get('CHISEL_CUTLASS_INCLUDE'):
            options.append(b'-DCHISEL_CUTLASS=1')
        if os.environ.get('CHISEL_ATTENTION_SHARED_STRIDE'):
            stride=int(os.environ['CHISEL_ATTENTION_SHARED_STRIDE'])
            if stride not in (128,136):raise ValueError('attention shared stride must be128 or136')
            options.append(('-DCHISEL_ATTENTION_SHARED_STRIDE='+str(stride)).encode())
        from .persistent_compile_cache import compile_key,store_binary
        import tempfile
        key=compile_key(source,options,include_dirs,_check(nvrtc.nvrtcVersion(),'NVRTC version'))
        cache_dir=Path(os.environ.get('CHISEL_NVRTC_CACHE_DIR',str(Path(tempfile.gettempdir())/f'chisel-nvrtc-{os.getuid()}')))
        cache_file=cache_dir/(key+'.cubin');self.compile_cache_key=key;self.compile_cache_hit=cache_file.is_file() and not os.environ.get('CHISEL_DUMP_PTX')
        if self.compile_cache_hit:cubin=cache_file.read_bytes()
        else:
            program=_check(nvrtc.nvrtcCreateProgram(source.encode(),b'persistent_training.cu',0,[],[]),'create NVRTC program')
            try:
                status=nvrtc.nvrtcCompileProgram(program,len(options),options)[0]
                if int(status):
                    length=_check(nvrtc.nvrtcGetProgramLogSize(program),'NVRTC log size')
                    log=bytearray(length)
                    _check(nvrtc.nvrtcGetProgramLog(program,log),'NVRTC log')
                    raise RuntimeError(log.decode())
                length=_check(nvrtc.nvrtcGetCUBINSize(program),'NVRTC cubin size')
                cubin=bytearray(length)
                _check(nvrtc.nvrtcGetCUBIN(program,cubin),'NVRTC cubin')
                if os.environ.get('CHISEL_DUMP_PTX'):
                    ptx=bytearray(_check(nvrtc.nvrtcGetPTXSize(program),'NVRTC PTX size'))
                    _check(nvrtc.nvrtcGetPTX(program,ptx),'NVRTC PTX')
                    Path(os.environ['CHISEL_DUMP_PTX']).write_bytes(ptx)
            finally:_check(nvrtc.nvrtcDestroyProgram(program),'destroy NVRTC program')
            store_binary(cache_file,cubin)
        self.module=_check(driver.cuModuleLoadData(bytes(cubin)),'load persistent module')
        self.function=_check(driver.cuModuleGetFunction(self.module,b'persistent_training'),'find persistent kernel')
        if self.dynamic_shared_bytes:
            _check(driver.cuFuncSetAttribute(self.function,driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,self.dynamic_shared_bytes),'opt in expanded shared workspace')
        occupancy=_check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.function,128,self.dynamic_shared_bytes),'cooperative occupancy')
        multiprocessors=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.maximum_resident_blocks=occupancy*multiprocessors
        multiplier=int(os.environ.get('CHISEL_PERSISTENT_BLOCK_MULTIPLIER','1'))
        self.blocks=min(blocks or multiprocessors*multiplier,self.maximum_resident_blocks)
        if self.blocks<1:raise RuntimeError('persistent kernel cannot reside on this GPU')
        attribute=driver.CUfunction_attribute
        self.registers=_check(driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,self.function),'register count')
        self.shared_bytes=_check(driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,self.function),'shared bytes')+self.dynamic_shared_bytes

    def launch(self, stages, cursor, step, clocks):
        if stages.device.index!=self.device:
            raise ValueError('persistent program and loaded kernel belong to different CUDA devices')
        _check(self.driver.cuCtxSetCurrent(self.context),'set persistent launch context')
        # Driver copies these scalar arguments while submitting the launch.
        args=[ctypes.c_uint64(stages.data_ptr()),ctypes.c_int(stages.shape[0]),
              ctypes.c_uint64(cursor.data_ptr()),ctypes.c_uint64(step.data_ptr()),ctypes.c_uint64(clocks.data_ptr())]
        pointers=(ctypes.c_void_p*len(args))(*(ctypes.addressof(arg) for arg in args))
        stream=self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(self.driver.cuLaunchCooperativeKernel(self.function,self.blocks,1,1,128,1,1,self.dynamic_shared_bytes,
                                                    stream,ctypes.addressof(pointers)),
               'launch persistent cooperative training kernel')


class PersistentMLPTrainer:
    """One launch per update of Linear->exact GELU->Linear, including Adam.

    Parameters are FP32 contiguous tensors, updated in place. X and cotangent
    are persistent input buffers; callers can copy new data between updates.
    The kernel owns first-order differentiation and optimizer state. Do not mix
    concurrent PyTorch autograd/optimizers with the same parameter storage.
    """
    def __init__(self, x, cotangent, weight1, bias1, weight2, bias2, *,
                 lr=2e-4, betas=(.9,.99), eps=1e-8, kernel=None, split_k=512):
        tensors=(x,cotangent,weight1,bias1,weight2,bias2)
        if any(t.device.type!='cuda' or t.device!=x.device or t.dtype!=torch.float32
               or not t.is_contiguous() for t in tensors):
            raise ValueError('persistent MLP requires same-device contiguous CUDA FP32 tensors')
        batch,inputs=x.shape
        hidden=weight1.shape[0];outputs=weight2.shape[0]
        if (weight1.shape!=(hidden,inputs) or weight2.shape!=(outputs,hidden)
            or bias1.shape!=(hidden,) or bias2.shape!=(outputs,) or cotangent.shape!=(batch,outputs)):
            raise ValueError('persistent MLP tensor shapes differ')
        if split_k<8 or split_k%8:raise ValueError('split_k must be a positive multiple of eight')
        self.kernel=kernel or CooperativeTrainingKernel()
        self.inputs=x;self.cotangent=cotangent
        self.parameters=(weight1,bias1,weight2,bias2)
        self.gradients=tuple(torch.empty_like(p) for p in self.parameters)
        self.first_moments=tuple(torch.zeros_like(p) for p in self.parameters)
        self.second_moments=tuple(torch.zeros_like(p) for p in self.parameters)
        self.hidden=torch.empty((batch,hidden),device=x.device)
        self.preactivation=torch.empty_like(self.hidden)
        self.output=torch.empty((batch,outputs),device=x.device)
        self.hidden_cotangent=torch.empty_like(self.hidden)
        self.input_cotangent=torch.empty_like(x)
        self.cursor=torch.zeros((),device=x.device,dtype=torch.int32)
        self.step_number=torch.zeros((),device=x.device,dtype=torch.int64)
        self.temporary=[]
        stages=[]
        def pointer(value):return 0 if value is None else value.data_ptr()
        def gemm(a,b,c,M,N,K,*,transpose_a=False,transpose_b=False,epilogue=0,bias=None,aux=None,split=False):
            parts=(K+split_k-1)//split_k if split else 1
            chunk=split_k if split else ((K+7)//8)*8
            target=c
            if parts>1:
                target=torch.empty((parts,M,N),device=x.device);self.temporary.append(target)
            stages.append([0,pointer(a),pointer(b),pointer(target),pointer(bias),pointer(aux),
                M,N,K,a.shape[-1],b.shape[-1],int(transpose_a)+2*int(transpose_b)+4*epilogue,parts,chunk,0,0])
            if parts>1:reduce(target,c,parts,M*N)
        def reduce(a,b,rows,cols):
            stages.append([1,pointer(a),pointer(b),0,0,0,rows,cols,0,0,0,0,0,0,0,0])
        def bias_gradient(a,b):
            rows,cols=a.shape;chunk=256;parts=(rows+chunk-1)//chunk
            partial=torch.empty((parts,cols),device=x.device);self.temporary.append(partial)
            stages.append([2,pointer(a),pointer(partial),0,0,0,rows,cols,chunk,0,0,0,0,0,0,0])
            reduce(partial,b,parts,cols)
        # All dependencies below are resolved by grid barriers inside one kernel.
        gemm(x,weight1,self.hidden,batch,hidden,inputs,transpose_b=True,epilogue=2,bias=bias1,aux=self.preactivation)
        gemm(self.hidden,weight2,self.output,batch,outputs,hidden,transpose_b=True,epilogue=1,bias=bias2)
        gemm(cotangent,self.hidden,self.gradients[2],outputs,hidden,batch,transpose_a=True,split=True)
        bias_gradient(cotangent,self.gradients[3])
        gemm(cotangent,weight2,self.hidden_cotangent,batch,hidden,outputs,epilogue=3,aux=self.preactivation)
        gemm(self.hidden_cotangent,x,self.gradients[0],hidden,inputs,batch,transpose_a=True,split=True)
        bias_gradient(self.hidden_cotangent,self.gradients[1])
        gemm(self.hidden_cotangent,weight1,self.input_cotangent,batch,inputs,hidden)
        bits=lambda value:struct.unpack('<I',struct.pack('<f',value))[0]
        for parameter,gradient,moment,variance in zip(self.parameters,self.gradients,self.first_moments,self.second_moments):
            stages.append([3,pointer(parameter),pointer(gradient),pointer(moment),pointer(variance),0,parameter.numel(),
                           bits(lr),bits(betas[0]),bits(betas[1]),bits(eps),0,0,0,0,0])
        self.stages=torch.tensor([stage+[0]*(24-len(stage)) for stage in stages],dtype=torch.int64,device=x.device)
        self.clocks=torch.empty(len(stages)+1,device=x.device,dtype=torch.int64)

    def step(self):
        self.kernel.launch(self.stages,self.cursor,self.step_number,self.clocks)
