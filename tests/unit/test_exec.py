from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pytest
from executor.vggish import vggish_input
from executor.vggish_audio_encoder import VggishAudioEncoder
from jina import Document, DocumentArray, Executor
from tensorflow.python.framework import ops


@pytest.fixture(scope="module")
def encoder() -> VggishAudioEncoder:
    ops.reset_default_graph()
    return VggishAudioEncoder()


@pytest.fixture(scope="module")
def gpu_encoder() -> VggishAudioEncoder:
    return VggishAudioEncoder(device='/GPU:0')


@pytest.fixture(scope='function')
def sample_file():
    return str(Path(__file__).parents[1] / 'test_data' / 'sample')


@pytest.fixture(scope='function')
def audio_sample_rate(sample_file):
    x_audio, sample_rate = librosa.load(f'{sample_file}.wav')
    return x_audio, sample_rate


@pytest.fixture(scope="function")
def nested_docs(audio_sample_rate) -> DocumentArray:
    audio, sample_rate = audio_sample_rate
    blob = vggish_input.waveform_to_examples(audio, sample_rate)
    docs = DocumentArray([Document(id="root1", blob=blob)])
    docs[0].chunks = [
        Document(id="chunk11", blob=blob),
        Document(id="chunk12", blob=blob),
        Document(id="chunk13", blob=blob),
    ]
    docs[0].chunks[0].chunks = [
        Document(id="chunk111", blob=blob),
        Document(id="chunk112", blob=blob),
    ]

    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert str(ex.vgg_model_path).endswith('vggish_model.ckpt')
    assert str(ex.pca_model_path).endswith('vggish_pca_params.ckpt')


def test_no_documents(encoder: VggishAudioEncoder):
    ops.reset_default_graph()
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: VggishAudioEncoder):
    ops.reset_default_graph()
    encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(encoder: VggishAudioEncoder):
    ops.reset_default_graph()
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_encode_single_document(audio_sample_rate):
    ops.reset_default_graph()
    x_audio, sample_rate = audio_sample_rate
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    model = VggishAudioEncoder()
    docs = DocumentArray([Document(blob=log_mel_examples)])
    model.encode(docs=docs)
    assert docs[0].embedding.shape == (128,)


def test_encode_multiple_documents(encoder: VggishAudioEncoder, audio_sample_rate):
    ops.reset_default_graph()
    x_audio, sample_rate = audio_sample_rate
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)

    docs = DocumentArray(
        [Document(blob=log_mel_examples), Document(blob=log_mel_examples)]
    )
    encoder.encode(docs, parameters={})
    assert docs[0].embedding.shape == (128,)
    assert docs[1].embedding.shape == (128,)


@pytest.mark.gpu
def test_encode_gpu(audio_sample_rate):
    x_audio, sample_rate = audio_sample_rate
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    model = VggishAudioEncoder(device='/GPU:0')
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (128,)


@pytest.mark.parametrize(
    "traversal_paths, counts",
    [
        [('c',), (('r', 0), ('c', 3), ('cc', 0))],
        [('cc',), (("r", 0), ('c', 0), ('cc', 2))],
        [('r',), (('r', 1), ('c', 0), ('cc', 0))],
        [('cc', 'r'), (('r', 1), ('c', 0), ('cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: Tuple[str],
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: VggishAudioEncoder,
):
    ops.reset_default_graph()
    encoder.encode(nested_docs, parameters={"traversal_paths": traversal_paths})
    for path, count in counts:
        embeddings = nested_docs.traverse_flat([path]).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count


@pytest.mark.parametrize('suffix', ['mp3', 'wav'])
def test_encode_wav_uri(sample_file, suffix):
    ops.reset_default_graph()
    model = VggishAudioEncoder(load_input_from='uri')
    fn = f'{sample_file}.{suffix}'
    doc = DocumentArray([Document(uri=fn)])
    model.encode(doc)
    assert doc[0].embedding.shape == (128,)


def test_encode_multiple_wav_uri(sample_file):
    ops.reset_default_graph()
    model = VggishAudioEncoder(load_input_from='uri')
    fn = f'{sample_file}.wav'
    broken_fn = f'{sample_file}'
    doc = DocumentArray([Document(uri=fn), Document(uri=broken_fn)])
    model.encode(doc)
    assert doc[0].embedding.shape == (128,)
    assert doc[1].embedding is None
