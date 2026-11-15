"""Immutable R2 objects with full readback verification; never remove local data."""
import configparser
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def client_from_config(config):
    import boto3
    from botocore.config import Config
    parser = configparser.ConfigParser()
    parser.read(config['credentials_file'])
    credentials = parser[config['credentials_section']]
    return boto3.client('s3', endpoint_url=credentials['endpoint'], region_name='auto',
        aws_access_key_id=credentials['access_key_id'], aws_secret_access_key=credentials['secret_access_key'],
        config=Config(signature_version='s3v4', connect_timeout=20, read_timeout=180,
                      retries={'max_attempts': 6}, max_pool_connections=config['workers'] * 2,
                      s3={'addressing_style': 'path'}))


def upload_verified(client, path, bucket, key, expected=None):
    from boto3.s3.transfer import TransferConfig
    path = Path(path)
    before = path.stat()
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'local checkpoint checksum mismatch: {path}')
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as error:
        if error.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
            raise
        head = None
    if head:
        if head['ContentLength'] != before.st_size or head.get('Metadata', {}).get('sha256') != digest:
            raise ValueError(f'immutable R2 key collision: {key}')
    else:
        client.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': digest}},
            Config=TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=2))
    response = client.get_object(Bucket=bucket, Key=key)
    actual, size = hashlib.sha256(), 0
    try:
        for chunk in iter(lambda: response['Body'].read(8 << 20), b''):
            actual.update(chunk)
            size += len(chunk)
    finally:
        response['Body'].close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'local file changed during upload: {path}')
    if size != before.st_size or actual.hexdigest() != digest:
        raise ValueError(f'R2 full-read verification failed: {key}')
    return dict(bucket=bucket, key=key, bytes=size, sha256=digest,
                verification='full_get_sha256', verified_unix=time.time())


def upload_receipt(client, root, receipt, bucket, prefix):
    root, receipt = Path(root).resolve(), Path(receipt).resolve()
    row = json.loads(receipt.read_text())
    path = (receipt.parent / row['file']).resolve()
    key = prefix.rstrip('/') + '/' + path.relative_to(root).as_posix()
    if path.stat().st_size != row['bytes']:
        raise ValueError(f'local checkpoint size mismatch: {path}')
    result = upload_verified(client, path, bucket, key, row['sha256'])
    upload_verified(client, receipt, bucket, prefix.rstrip('/') + '/' + receipt.relative_to(root).as_posix())
    marker = receipt.with_name(receipt.name.replace('.ready.json', '.uploaded.json'))
    atomic_json(marker, result)
    return result


if __name__ == '__main__':
    import argparse
    import yaml
    from chisel.harness import resolve
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    config = resolve(parser.parse_args().config)
    client = client_from_config(config)
    rows = [upload_verified(client, item['path'], config['bucket'], item['key'], item['sha256'])
            for item in config['files']]
    output = Path(config['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(dict(files=rows), sort_keys=False))
    temporary.replace(output)
