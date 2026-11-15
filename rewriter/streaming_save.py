"""Sequential torch.save transport with a SHA256 over exactly the written bytes.

This helper changes no serialization format. Seeking, short writes and external
position changes fail closed, so a digest can never silently omit overwritten data.
It is not enabled in any capture source merely by importing this module.
"""
import hashlib


class HashingWriter:
    def __init__(self,stream):
        if stream.tell()!=0:raise ValueError('hashing requires a fresh zero-offset stream')
        self.stream=stream;self.position=0;self.hash=hashlib.sha256()

    def write(self,data):
        if self.stream.tell()!=self.position:raise OSError('underlying stream moved outside hashing writer')
        view=memoryview(data).cast('B')
        written=self.stream.write(view)
        if written!=len(view):raise OSError('short write; digest is not publishable')
        self.hash.update(view);self.position+=written
        return written

    def tell(self):return self.position
    def seek(self,*args):raise OSError('hashing writer forbids seek/overwrite')
    def flush(self):return self.stream.flush()
    def hexdigest(self):return self.hash.hexdigest()
