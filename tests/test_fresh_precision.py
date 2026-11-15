import copy
import pytest
from rewriter.gpt_precision import precision_metadata
from scripts.evaluate_chisel_fresh import checkpoint_gpt_precision


def test_fresh_evaluation_preserves_historical_baseline():
    assert checkpoint_gpt_precision({'metadata':{'options':{}}})=='baseline'


def test_fresh_evaluation_requires_explicit_fp32_arithmetic_binding():
    saved={'metadata':{'options':{'gpt_precision':'fp32'},
                       'gpt_precision':precision_metadata('fp32')}}
    original=copy.deepcopy(saved)
    assert checkpoint_gpt_precision(saved)=='fp32'
    assert saved==original
    for key,value in [('mode','baseline'),('rms_epsilon',1e-7),('tf32',True),
                      ('attention_backend','flash'),('residual_dtype','bfloat16')]:
        wrong=copy.deepcopy(saved);wrong['metadata']['gpt_precision'][key]=value
        with pytest.raises(ValueError,match='precision/normalization'):
            checkpoint_gpt_precision(wrong)
    del saved['metadata']['gpt_precision']
    with pytest.raises(ValueError,match='precision/normalization'):
        checkpoint_gpt_precision(saved)


def test_fresh_evaluation_rejects_conflicting_mode_and_metadata():
    saved={'metadata':{'options':{},'gpt_precision':precision_metadata('fp32')}}
    with pytest.raises(ValueError,match='precision/normalization'):
        checkpoint_gpt_precision(saved)
