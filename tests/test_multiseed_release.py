import json,sys
import pytest
from scripts import finish_multiseed_leases as release


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'D',tmp_path)
    monkeypatch.setattr(sys,'argv',['release','--execute'])
    (tmp_path/'plan.json').write_text(json.dumps(dict(lanes=[dict(gpu='owned-node',seeds=[1])])) )
    (tmp_path/'lane1-state.json').write_text(json.dumps(dict(status='complete',gpu='owned-node',completed=[1],teacher_completed=[1])))
    (tmp_path/'seed1-archive.json').write_text(json.dumps(dict(seed=1,states=1000)))
    (tmp_path/'seed1-teacher-archive.json').write_text(json.dumps(dict(seed=1,states=1000,panels=2000)))


def test_release_only_own_completed_verified_lane(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);calls=[]
    def cli(*args):
        calls.append(args)
        return '' if args[1]=='exec' else '{"status":"done"}'
    monkeypatch.setattr(release,'cli',cli);release.main()
    assert [c[1] for c in calls]==['exec','down']
    assert all(c[2]=='owned-node' for c in calls)
    assert json.loads((tmp_path/'lane1-release.json').read_text())['status']=='released'


def test_explicit_handoff_never_releases(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    (tmp_path/'hold-lane1.json').write_text('{"owner":"next-study"}')
    monkeypatch.setattr(release,'cli',lambda *a:pytest.fail('handoff should not touch GPU'))
    release.main();assert json.loads((tmp_path/'lane1-release.json').read_text())['status']=='handoff_held'


def test_active_or_incomplete_evidence_fails_closed(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(release,'cli',lambda *a:calls.append(a) or '12345')
    with pytest.raises(RuntimeError):release.main()
    assert len(calls)==1 and calls[0][1]=='exec'
    (tmp_path/'seed1-teacher-archive.json').write_text('{"seed":1,"states":999,"panels":1998}')
    calls.clear()
    with pytest.raises(AssertionError):release.main()
    assert not calls
