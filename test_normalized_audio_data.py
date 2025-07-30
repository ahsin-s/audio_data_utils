import pytest
import tempfile
import numpy as np
import soundfile as sf
from normalized_audio_data import chunk_audio_data 


def test_chunk_audio_data():
    
    audio_data = np.ones(100)
    with tempfile.TemporaryDirectory() as tempdir:
        tempoutputpath = tempdir + "/output.wav"
        result = chunk_audio_data(audio_data, 10, 10, tempoutputpath, overlap_factor=0.1)
        assert len(result) == 11

        # open each result and check it has the right length and samplerate 
        for res in result:
            data, sr = sf.read(res)
            assert len(data) == 10 
            assert sr == 10
    
