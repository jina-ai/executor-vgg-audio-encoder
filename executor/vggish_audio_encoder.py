__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests as _requests
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

from .vggish.vggish_params import INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME
from .vggish.vggish_postprocess import Postprocessor
from .vggish.vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint
from .vggish.vggish_input import wavfile_to_examples, mp3file_to_examples

tf.compat.v1.disable_eager_execution()

cur_dir = os.path.dirname(os.path.abspath(__file__))


class VggishAudioEncoder(Executor):
    """
    Encode audio data with Vggish embeddings
    """

    def __init__(
        self,
        model_path: str = Path(cur_dir) / 'models',
        load_input_from: str = 'blob',
        device: str = '/CPU:0',
        traversal_paths: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param model_path: path of the models directory. The directory should contain
            'vggish_model.ckpt' and 'vggish_pca_params.ckpt'. Setting this to a new directory
            will download the files.
        :param load_input_from: the place where input data is stored, either 'uri' or 'blob'.
            When set to 'uri', the model reads wave file (.mp3 or .wav) from the file path specified by the 'uri'.
            When set to 'blob', the model reads wave form array from the `blob` attribute.
        :param device: device to run the model on e.g. '/CPU:0','/GPU:0','/GPU:2'
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        :param traversal_paths: fallback batch size in case there is not
            batch size sent in the request
        """

        super().__init__(*args, **kwargs)
        self.traversal_paths = traversal_paths or ['r']
        self.logger = JinaLogger(self.__class__.__name__)
        self.device = device
        if load_input_from not in ('uri', 'blob'):
            self.logger.warning(f'unknown setting to load_input_form. Set to default value, load_input_from="blob"')
            load_input_from = 'blob'
        self._input = load_input_from
        self.model_path = Path(model_path)
        self.vgg_model_path = self.model_path / 'vggish_model.ckpt'
        self.pca_model_path = self.model_path / 'vggish_pca_params.ckpt'
        self.model_path.mkdir(
            exist_ok=True
        )  # Create the model directory if it does not exist yet

        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if 'GPU' in device:
            gpu_index = 0 if 'GPU:' not in device else int(device.split(':')[-1])
            if len(gpus) < gpu_index + 1:
                raise RuntimeError(f'Device {device} not found on your system!')
            cpus.append(gpus[gpu_index])
        tf.config.experimental.set_visible_devices(devices=cpus)

        if not self.vgg_model_path.exists():
            self.logger.info(
                'VGGish model cannot be found from the given model path, '
                'downloading a new one...'
            )
            try:
                r = _requests.get(
                    'https://storage.googleapis.com/audioset/vggish_model.ckpt'
                )
                r.raise_for_status()
            except _requests.exceptions.HTTPError:
                self.logger.error(
                    'received HTTP error response, cannot download vggish model'
                )
                raise
            except _requests.exceptions.RequestException:
                self.logger.error('Connection error, cannot download vggish model')
                raise

            with open(self.vgg_model_path, 'wb') as f:
                f.write(r.content)

        if not self.pca_model_path.exists():
            self.logger.info(
                'PCA model cannot be found from the given model path, '
                'downloading a new one...'
            )
            try:
                r = _requests.get(
                    'https://storage.googleapis.com/audioset/vggish_pca_params.npz'
                )
                r.raise_for_status()
            except _requests.exceptions.HTTPError:
                self.logger.error(
                    'received HTTP error response, cannot download pca model'
                )
                raise
            except _requests.exceptions.RequestException:
                self.logger.error('Connection error, cannot download pca model')
                raise

            with open(self.pca_model_path, 'wb') as f:
                f.write(r.content)

        self.sess = tf.compat.v1.Session()
        define_vggish_slim()
        load_vggish_slim_checkpoint(self.sess, str(self.vgg_model_path))
        self.feature_tensor = self.sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
        self.post_processor = Postprocessor(str(self.pca_model_path))
        self.batch_size = batch_size

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict = {}, **kwargs):
        """
        Compute embeddings and store them in the `docs` array.

        :param docs: documents sent to the encoder. The docs must have `text`.
            By default, the input `text` must be a `list` of `str`.
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example, `parameters={'traversal_paths': ['r'],
            'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        :return:
        """
        if not docs:
            return

        document_batches_generator = docs.batch(
            traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
            batch_size=parameters.get('batch_size', self.batch_size),
            require_attr=self._input,
        )

        for batch_docs in document_batches_generator:
            mel_list = []
            blob_shape_list = []
            if self._input == 'uri':
                for doc in batch_docs:
                    f_suffix = Path(doc.uri).suffix
                    if f_suffix == '.wav':
                        blob = wavfile_to_examples(doc.uri)
                    elif f_suffix == '.mp3':
                        blob = mp3file_to_examples(doc.uri)
                    else:
                        self.logger.warning(f'unsupported format {f_suffix}. Please use .mp3 or .wav')
                        self.logger.warning(f'skip {doc.uri}')
                        blob_shape_list.append(0)
                        continue
                    mel_list.append(blob)
                    blob_shape_list.append(blob.shape[0])
            elif self._input == 'blob':
                mel_list = batch_docs.get_attributes('blob')
                blob_shape_list = [blob.shape[0] for blob in mel_list]
            try:
                mel_array = np.vstack(mel_list)
            except ValueError as e:
                self.logger.error(f'the blob must have the same size, {e}')
                continue
            [embeddings] = self.sess.run(
                [self.embedding_tensor],
                feed_dict={self.feature_tensor: mel_array}
            )
            result = self.post_processor.postprocess(embeddings)
            beg = 0
            for doc, blob_shape in zip(batch_docs, blob_shape_list):
                if blob_shape == 0:
                    continue
                emb = result[beg:beg+blob_shape, :]
                beg += blob_shape
                # convert the embedding to [-1, 1]
                doc.embedding = np.mean((np.float32(emb) - 128.0) / 128.0, axis=0)

    def close(self):
        self.sess.close()
