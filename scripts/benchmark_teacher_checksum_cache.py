"""Measure immutable teacher-file verification only; no GPT or training timing."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_cpu_norm_preservation import _TEACHER_CHECKSUM_CACHE, _verify_teacher_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = []
    for directory in sorted(args.teacher_root.glob('teacher_state*')):
        manifest = json.loads((directory / 'manifest.json').read_text())
        files.extend((directory / row['file'], row['sha256']) for row in manifest['files'])
    if not files or len(files) > 128:
        raise ValueError('Need 1..128 files fitting the bounded checksum memo')
    _TEACHER_CHECKSUM_CACHE.clear()
    measurements = []
    for label in ('first_verification', 'unchanged_file_reuse'):
        counts = dict(hits=0, files_hashed=0, bytes_hashed=0, hash_seconds=0.)
        started = time.perf_counter()
        for path, expected in files:
            _verify_teacher_file(path, expected, counts)
        measurements.append(dict(label=label, elapsed_seconds=time.perf_counter()-started, **counts))
    if measurements[0]['files_hashed'] != len(files) or measurements[1]['hits'] != len(files):
        raise RuntimeError('Unexpected misses or file changes during measurement')
    result = dict(scope=__doc__, files=len(files), measurements=measurements,
                  limitations='One measurement per path, on a shared CPU host during a separate fit; this is verification overhead only, not end-to-end speedup. Files are required to stay immutable. No tensors cached.',
                  gpt_forwards=0, gpt_backwards=0, optimizer_steps=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
