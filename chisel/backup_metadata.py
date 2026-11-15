"""Back up run recipes, source archives and metrics; trajectory shards use the verified uploader."""
import argparse
import configparser
import hashlib
from pathlib import Path
import time
import boto3
from chisel.harness import resolve


def run(config):
    credentials = configparser.ConfigParser()
    credentials.read(config['credentials_file'])
    values = credentials[config['credentials_section']]
    client = boto3.client('s3', endpoint_url=values['endpoint'], region_name='auto',
                          aws_access_key_id=values['access_key_id'], aws_secret_access_key=values['secret_access_key'])
    root, seen = Path(config['root']), {}
    for _ in range(config['polls']):
        for path in root.rglob('*'):
            if not path.is_file() or not any(path.name.endswith(suffix) for suffix in config['suffixes']):
                continue
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if seen.get(path) == stamp:
                continue
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            key = config['prefix'].rstrip('/') + '/' + path.relative_to(root).as_posix()
            try:
                client.put_object(Bucket=config['bucket'], Key=key, Body=body, Metadata={'sha256': digest})
                response = client.get_object(Bucket=config['bucket'], Key=key)
                try:
                    actual = hashlib.sha256(response['Body'].read()).hexdigest()
                finally:
                    response['Body'].close()
                if actual != digest:
                    raise IOError(f'R2 metadata verification failed: {key}')
                seen[path] = stamp
            except Exception as error:
                print(f'{path}: {type(error).__name__}: {error}', flush=True)
        print(f'verified metadata files: {len(seen)}', flush=True)
        time.sleep(config['poll_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    run(resolve(parser.parse_args().config))
