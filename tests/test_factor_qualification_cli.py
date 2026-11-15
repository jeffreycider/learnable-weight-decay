import sys
import pytest
from scripts import qualify_cpu_factor_panel as cpu
from scripts import qualify_gpu_factor_panel as gpu


def cpu_args(tmp_path):
    return ['qualify','--backend','direct','--checkpoint','missing.pt','--output',str(tmp_path/'out'),'--teacher-cache-root','missing-cache']


def gpu_args(tmp_path):
    return ['qualify','--checkpoint','missing.pt','--state','missing-state','--panel','missing-panel','--panel-metadata','missing-json','--output',str(tmp_path/'out')]


def test_cpu_streamed_norm_rejects_dense_before_io(monkeypatch,tmp_path):
    args=cpu_args(tmp_path);args[2]='dense'
    monkeypatch.setattr(sys,'argv',args+['--streamed-norm','--private-eps','1e-8'])
    with pytest.raises(ValueError,match='requires direct'):cpu.main()
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('module,args_fn',[(cpu,cpu_args),(gpu,gpu_args)])
@pytest.mark.parametrize('rows',['0','2049'])
def test_invalid_norm_chunks_rejected_before_io_or_cuda(module,args_fn,rows,monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',args_fn(tmp_path)+['--streamed-norm','--norm-chunk-rows',rows])
    with pytest.raises(ValueError,match='norm chunk'):module.main()
    assert not (tmp_path/'out').exists()


def test_gpu_streamed_norm_cli_accepted_before_explicit_cuda_guard(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',gpu_args(tmp_path)+['--streamed-norm','--norm-chunk-rows','7'])
    monkeypatch.setattr(gpu.torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA required'):gpu.main()
    assert not (tmp_path/'out').exists()


def test_cpu_streamed_norm_cli_keeps_default_epsilon_and_accepts_explicit_override(monkeypatch,tmp_path):
    # Stop at the first post-validation setup operation; never load a model.
    class StopSetup(Exception):pass
    parsed=[];original=cpu.argparse.ArgumentParser.parse_args
    def parse(parser,*args,**kwargs):
        result=original(parser,*args,**kwargs);parsed.append(result);return result
    def stop(*args):raise StopSetup
    monkeypatch.setattr(cpu.argparse.ArgumentParser,'parse_args',parse)
    monkeypatch.setattr(cpu.torch,'set_num_threads',stop)
    for index,extra in enumerate(([],['--private-eps','1e-8'])):
        args=cpu_args(tmp_path);args[6]=str(tmp_path/f'out{index}')
        monkeypatch.setattr(sys,'argv',args+['--streamed-norm']+extra)
        with pytest.raises(StopSetup):cpu.main()
    assert [a.private_eps for a in parsed]==[1e-12,1e-8]
    assert all(a.streamed_norm and a.norm_chunk_rows==128 for a in parsed)


def test_imported_streamed_norm_source_is_pinned(tmp_path):
    from pathlib import Path
    digests=cpu.pin_sources(tmp_path)
    name='rewriter/experimental_factor_norm.py'
    assert name in digests
    assert (tmp_path/'sources'/name).read_bytes()==Path(name).read_bytes()
