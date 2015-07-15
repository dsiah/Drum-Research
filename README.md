## Drum Research


### Dependencies 
You will need pip installed on your machine. (https://pip.pypa.io/en/latest/installing.html)[Installing Python Install Package instructions]

Run these two commands to get the IPython library bundled with IPython Notebooks as well as Librosa.

```

pip install librosa
pip install "ipython[notebook]"

```

### File Structure

```

Resources/
|---- Sample
|---- More samples ...

data/
|---- Onset STFT CSV
|---- More onset STFT CSVs ...

libs/
|---- Python modules 
|---- More python modules ...

LVQLib.ipynb
SampleExtraction.ipynb

```

- LVQLib.ipynb : A python notebook that outlines the LVQ neural network
- SampleExtraction.ipynb : Sample Extraction Pipeline gets the onsets and saves them
- data/ : where the extracted CSV STFT (Short Time Fourier Transform) of onsets reside
- Resources/ : Samples for testing the SVQNet
- libs/ : Python modules converted from the Ipython notebooks so they can be used elsewhere