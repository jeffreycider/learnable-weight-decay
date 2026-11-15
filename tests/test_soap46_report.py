from scripts.report_soap46_overnight import comparisons,MATCH


def test_report_pairs_only_same_seed_variant_and_numerical_source():
    config={key:'same' for key in MATCH}
    config.update(variant='outer-only',seed=1,base_lr=.03,cwd=.0125)
    learned={'name':'learned','frozen':False,'config':config}
    frozen={'name':'frozen','frozen':True,'config':{**config,'base_lr':.0375,'cwd':.025}}
    assert comparisons([learned,frozen])==[(learned,frozen)]
    for key in ('seed','variant','implementation_sha256','source_sha256'):
        wrong={**frozen,'config':{**frozen['config'],key:'different'}}
        assert comparisons([learned,wrong])==[]


def test_report_refuses_ambiguous_repeated_controls():
    import pytest
    config={key:'same' for key in MATCH};config.update(base_lr=.0375,cwd=.025)
    learned={'name':'learned','frozen':False,'config':config}
    frozen={'name':'frozen','frozen':True,'config':config}
    with pytest.raises(RuntimeError,match='Ambiguous'):
        comparisons([learned,frozen,{**frozen,'name':'another'}])


def test_same_base_diagnostic_is_distinct_from_published_reference():
    from scripts.report_soap46_overnight import same_base_comparisons
    config={key:'same' for key in MATCH};config.update(seed=1,base_lr=.03,cwd=.025)
    learned={'name':'learned','frozen':False,'config':config}
    same={'name':'same-base','frozen':True,'config':config}
    published={'name':'published','frozen':True,'config':{**config,'base_lr':.0375}}
    assert comparisons([learned,same,published])==[(learned,published)]
    assert same_base_comparisons([learned,same,published])==[(learned,same)]
