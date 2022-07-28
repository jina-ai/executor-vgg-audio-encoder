__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests as _requests
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

from .vggish.vggish_params import INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME
from .vggish.vggish_postprocess import Postprocessor
from .vggish.vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint
from .vggish.vggish_input import wavfile_to_examples, mp3file_to_examples, waveform_to_examples

import warnings

tf.compat.v1.disable_eager_execution()

cur_dir = os.path.dirname(os.path.abspath(__file__))


class VggishAudioEncoder(Executor):
    """
    Encode audio data with Vggish embeddings
    """

    def __init__(
        self,
        model_path: str = Path(cur_dir).parents[0] / 'models',
        load_input_from: str = 'uri',
        min_duration: int = 10,
        device: str = '/CPU:0',
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param model_path: path of the models directory. The directory should contain
            'vggish_model.ckpt' and 'vggish_pca_params.ckpt'. Setting this to a new directory
            will download the files.
        :param load_input_from: the place where input data is stored, either 'uri', 'log_mel', 'waveform'.
            When set to 'uri', the model reads wave file (.mp3 or .wav) from the file path specified by the 'uri'.
            When set to 'log_mel', the model reads log melspectrogram array from the `blob` attribute.
            When set to 'waveform', the model reads wave form array from the `blob` attribute. This requires the sample rate information stored at `.tags['sample_rate']` as `float`.
        :param min_duration: the minimal duration of the audio data in seconds. The input data will not be encoded if it is shorter than this duration.
        :param device: device to run the model on e.g. '/CPU:0','/GPU:0','/GPU:2'
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        :param access_paths: fallback batch size in case there is not
            batch size sent in the request
        :param traversal_paths: please use access_paths
        """

        super().__init__(*args, **kwargs)
        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths
        self.logger = JinaLogger(self.__class__.__name__)
        self.device = device
        self.min_duration = min_duration
        if load_input_from not in ('uri', 'log_mel', 'waveform'):
            self.logger.warning(f'unknown setting to load_input_form. Set to default value, load_input_from="uri"')
            load_input_from = 'uri'
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
    def encode(self, docs: DocumentArray, parameters: dict = {}, **kwargs):
        """
        Compute embeddings and store them in the `docs` array.

        :param docs: documents sent to the encoder. The docs must have `text`.
            By default, the input `text` must be a `list` of `str`.
        :param parameters: dictionary to define the `access_paths` and the
            `batch_size`. For example, `parameters={'access_paths': ['r'],
            'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        :return:
        """

        document_batches_generator = DocumentArray(
            docs[parameters.get('access_paths', self.access_paths)],
        ).batch(batch_size=parameters.get('batch_size', self.batch_size))

        for batch_docs in document_batches_generator:
            tensor_shape_list, mel_list = self._get_input_feature(batch_docs)
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
            for doc, tensor_shape in zip(batch_docs, tensor_shape_list):
                if tensor_shape == 0:
                    continue
                emb = result[beg:beg+tensor_shape, :]
                beg += tensor_shape
                doc.embedding = np.float32(emb[:self.min_duration]).flatten()

    def _get_input_feature(self, batch_docs):
        mel_list = []
        tensor_shape_list = []
        if self._input == 'uri':
            for doc in batch_docs:
                f_suffix = Path(doc.uri).suffix
                if f_suffix == '.wav':
                    tensor = wavfile_to_examples(doc.uri)
                elif f_suffix == '.mp3':
                    tensor = mp3file_to_examples(doc.uri)
                else:
                    self.logger.warning(f'unsupported format {f_suffix}. Please use .mp3 or .wav')
                    self.logger.warning(f'skip {doc.uri}')
                    tensor_shape_list.append(0)
                    continue
                if tensor.shape[0] < self.min_duration:
                    tensor_shape_list.append(0)
                    continue
                mel_list.append(tensor)
                tensor_shape_list.append(tensor.shape[0])
        elif self._input == 'waveform':
            for doc in batch_docs:
                data = doc.tensor
                sr = doc.tags['sample_rate']
                if len(data.shape) > 1:
                    data = np.mean(data, axis=0)
                samples = data / 32768.0  # Convert to [-1.0, +1.0]
                tensor = waveform_to_examples(samples, sr)
                if tensor.shape[0] < self.min_duration:
                    tensor_shape_list.append(0)
                    continue
                tensor_shape_list.append(tensor.shape[0])
                mel_list.append(tensor)
        elif self._input == 'log_mel':
            _mel_list = batch_docs.tensors
            for tensor in _mel_list:
                if tensor.shape[0] < self.min_duration:
                    tensor_shape_list.append(0)
                    continue
                else:
                    tensor_shape_list.append(tensor.shape[0])
                mel_list.append(tensor)
        return tensor_shape_list, mel_list

    def close(self):
        self.sess.close()
