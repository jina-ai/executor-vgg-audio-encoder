import glob
from pathlib import Path
from typing import Tuple

import librosa
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
def audio_sample_rate(sample_file):
    x_audio, sample_rate = librosa.load(f'{sample_file}.wav')
    return x_audio, sample_rate


@pytest.fixture(scope="function")
def nested_docs(audio_sample_rate, sample_file) -> DocumentArray:
    fn = f'{sample_file}.wav'
    docs = DocumentArray([Document(id="root1", uri=fn)])
    docs[0].chunks = [
        Document(id=f'chunk1{i}', uri=fn) for i in range(3)]
    docs[0].chunks[0].chunks = [
        Document(id=f'chunk11{i}', uri=fn) for i in range(2)]
    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert str(ex.vgg_model_path).endswith('vggish_model.ckpt')
    assert str(ex.pca_model_path).endswith('vggish_pca_params.ckpt')


def test_no_documents(encoder: VggishAudioEncoder):
    ops.reset_default_graph()
    docs = DocumentArray()
    encoder.encode(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_docs_no_blobs(encoder: VggishAudioEncoder):
    ops.reset_default_graph()
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray())
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_encode_log_mel(audio_sample_rate):
    ops.reset_default_graph()
    model = VggishAudioEncoder(load_input_from='log_mel')
    audio, sample_rate = audio_sample_rate
    blob = vggish_input.waveform_to_examples(audio, sample_rate)
    docs = DocumentArray([Document(blob=blob)])
    model.encode(docs=docs)
    assert docs[0].embedding.shape == (1280,)


@pytest.mark.parametrize('num_docs', [4, 16, 128])
def test_encode_multiple_documents(encoder: VggishAudioEncoder, sample_file, num_docs):
    ops.reset_default_graph()
    fn = f'{sample_file}.wav'
    docs = DocumentArray([Document(uri=fn) for _ in range(num_docs)])
    encoder.encode(docs, parameters={})
    for doc in docs:
        assert doc.embedding.shape == (1280,)


@pytest.mark.gpu
def test_encode_gpu(audio_sample_rate):
    x_audio, sample_rate = audio_sample_rate
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])
    model = VggishAudioEncoder(device='/GPU:0')
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (1280,)


@pytest.mark.parametrize(
    "traversal_paths, counts",
    [
        ['@c', (('@r', 0), ('@c', 3), ('@cc', 0))],
        ['@cc', (('@r', 0), ('@c', 0), ('@cc', 2))],
        ['@r', (('@r', 1), ('@c', 0), ('@cc', 0))],
        ['@cc,r', (('@r', 1), ('@c', 0), ('@cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: str,
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: VggishAudioEncoder,
):
    ops.reset_default_graph()
    encoder.encode(nested_docs, parameters={'traversal_paths': traversal_paths})
    for path, count in counts:
        embeddings = nested_docs[path].embeddings
        if count != 0:
            assert len([em for em in embeddings if em is not None]) == count
        else:
            assert embeddings is None


@pytest.mark.parametrize('suffix', ['mp3', 'wav'])
def test_encode_uri(encoder, sample_file, suffix):
    ops.reset_default_graph()
    fn = f'{sample_file}.{suffix}'
    doc = DocumentArray([Document(uri=fn)])
    encoder.encode(doc)
    assert doc[0].embedding.shape == (1280,)


def test_encode_broken_uri(encoder, sample_file):
    ops.reset_default_graph()
    fn = f'{sample_file}.wav'
    broken_fn = f'{sample_file}'
    doc = DocumentArray([Document(uri=fn), Document(uri=broken_fn)])
    encoder.encode(doc)
    assert doc[0].embedding.shape == (1280,)
    assert doc[1].embedding is None


def test_encode_waveform(audio_sample_rate):
    ops.reset_default_graph()
    x_audio, sample_rate = audio_sample_rate
    model = VggishAudioEncoder(load_input_from='waveform')
    docs = DocumentArray([Document(blob=x_audio, tags={'sample_rate': sample_rate})])
    model.encode(docs=docs)
    assert docs[0].embedding.shape == (1280,)


def test_embedding(encoder):
    def get_label(uri):
        return uri.split('.')[0].split('_')[-1]
    audioset_path = Path(__file__).parents[1] / 'test_data' / 'audioset'
    docs = DocumentArray()
    for fn in glob.glob(f'{audioset_path}/*.mp3'):
        doc = Document(uri=fn)
        docs.append(doc)
    encoder.encode(docs=docs)
    q_docs = docs
    q_docs.match(docs)
    for d in q_docs:
        q_label = get_label(d.uri)
        if q_label == 'airplane':
            for m in d.matches[:2]:
                print(f'{m.scores["cosine"].value}')
                assert q_label == get_label(m.uri)


@pytest.mark.parametrize(('min_duration', 'is_none'), [[5, False], [15, True]])
def test_min_duration(min_duration, is_none):
    ops.reset_default_graph()
    encoder = VggishAudioEncoder(min_duration=min_duration)
    audioset_path = Path(__file__).parents[1] / 'test_data' / 'audioset'
    docs = DocumentArray()
    for fn in glob.glob(f'{audioset_path}/*.mp3'):
        doc = Document(uri=fn)
        docs.append(doc)
    encoder.encode(docs=docs)
    for d in docs:
        assert (d.embedding is None) == is_none
