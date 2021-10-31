# VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model for generating embeddings for audio data. 

The input Document is expected to have 
- Either `blob` attribute storing the log mel spectrogram features stored
- Or `uri` attribute storing the file path information of a `.mp3` or `.wav` file

You need to set `load_input_from` argument in the `init()` function to specify from where the audio data is stored. By default, it is set to `blob`.

With the Vggish model, VggishAudioEncoder encode the audio data into a 128 dimensional vector and stored in the 
`embedding` attribute.

For more information, such as run executor on gpu, check out [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
