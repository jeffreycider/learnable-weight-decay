"""Content-addressed NVRTC binaries; every include file participates in the key."""
import hashlib
from pathlib import Path
import os
import tempfile


def compile_key(source,options,include_dirs,compiler_version):
    digest=hashlib.sha256()
    def add(value):
        value=value if isinstance(value,bytes) else str(value).encode()
        digest.update(len(value).to_bytes(8,'little'));digest.update(value)
    add(source);add(tuple(compiler_version))
    for option in options:add(option)
    for directory in include_dirs:
        directory=Path(directory);add(directory.resolve())
        for path in sorted(directory.rglob('*')):
            if path.is_file():add(path.relative_to(directory));add(path.read_bytes())
    return digest.hexdigest()


def store_binary(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
