from rewriter.persistent_compile_cache import compile_key,store_binary


def test_cache_identity_covers_headers_options_and_compiler(tmp_path):
    header=tmp_path/'nested'/'input.cuh';header.parent.mkdir();header.write_text('first')
    def key(options=(b'--fmad=false',),version=(12,8)):
        return compile_key('source',options,[tmp_path],version)
    original=key();assert original==key()
    header.write_text('second');assert original!=key()
    header.write_text('first');assert original==key()
    assert original!=key(options=(b'--fmad=true',))
    assert original!=key(version=(12,9))


def test_atomic_binary_roundtrip(tmp_path):
    target=tmp_path/'nested'/'cache.cubin';store_binary(target,b'first')
    store_binary(target,b'second');assert target.read_bytes()==b'second'
    assert list(target.parent.iterdir())==[target]
