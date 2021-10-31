# VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 

The input audio data is expected to be 
- the log mel spectrogram features stored in `blob` attribute
- the waveform data stored in `blob` attribute together with the sampling rate information stored in `tags['sample_rate']` as a float number
- the file path information of a `.mp3` or `.wav` file stored in the `uri` attribute

You need to set `load_input_from` argument in the `init()` function to specify from where the audio data is stored. By default, it is set to `uri`.
- `log_mel` to use the log mel spectrogram features 
- `waveform` to use the waveform data 
- `uri` to use the file path information

With the Vggish model, `VggishAudioEncoder` encodes the audio data into a 128 dimensional vector and stored in the 
`embedding` attribute.

For more information, such as run executor on gpu, check out [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
