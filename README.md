## Drum Research


### Dependencies 
You will need pip installed on your machine. [Installing Python Install Package instructions](https://pip.pypa.io/en/latest/installing.html)

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

Data/
|---- Onset STFT CSV
|---- More onset STFT CSVs ...

Libs/
|---- Python modules 
|---- More python modules ...

LVQLib.ipynb
SampleExtraction.ipynb

```

- __LVQLib.ipynb__ : A python notebook that outlines the LVQ neural network
- __SampleExtraction.ipynb__ : Sample Extraction Pipeline gets the onsets and saves them
- __data/__ : where the extracted CSV STFT (Short Time Fourier Transform) of onsets reside
- __Resources/__ : Samples for testing the SVQNet
- __libs/__ : Python modules converted from the Ipython notebooks so they can be used elsewhere
