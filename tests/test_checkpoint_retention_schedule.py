from baseline.train import checkpoint_due


def test_regular_checkpoint_schedule_is_identical_without_explicit_retention():
    for cadence in (0,1,25,125,250):
        config=dict(max_steps=1250,checkpoint_every=cadence)
        for step in range(1000,1251):
            old=bool(cadence and step%cadence==0 and step<1250)
            assert checkpoint_due(config,step)==old


def test_explicit_actions_save_off_cadence_and_terminal_uses_final_checkpoint():
    config=dict(max_steps=1250,checkpoint_every=50,preserve_checkpoint_steps=[1001,1125,1250])
    assert [s for s in range(1001,1251) if checkpoint_due(config,s)]==[1001,1050,1100,1125,1150,1200]
    config['checkpoint_every']=0
    assert [s for s in range(1001,1251) if checkpoint_due(config,s)]==[1001,1125]
