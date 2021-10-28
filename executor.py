from jina import Executor, DocumentArray, requests


class VGGishAudioEncoder(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
