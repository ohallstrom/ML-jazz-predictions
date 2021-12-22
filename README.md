# ML-jazz-predictions
### Predicting the next chord in Jazz music through Machine Learning
By training a LSTM on data from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html), we are in this project investigating if the lead melody, in addition to the previously played chords, contains information that helps the prediction of the next chord.


### External libraries
To install the required libraries, in CLI, run: `pip install -r requirements.txt`

### File Structure 
- Data folder contains the preprocessed data in csv format, the predictions of the models as well as the wjazzdb dataset.
- src - Contains all the python files for training, testing, gridsearch and preprocessing 
	- logs : Contains the log files for each model. Files generated during train are placed in the folder correponding to the model
	- models : Contains the saved pretrained models. We save the model with the best validation accuracy during training.
	- utils : Contains the helper functions for preprocessing, training and testing
	- data.py : Code to load data during train/test
	- test_chord_mappings.py : Code to test chord mappings.
	- preprocess_data.py: Code to generate preprocessed data from raw wjazzdb tables. Generated data is in the data folder
	- models.py : Code for the ChordSequence Model
	- gridsearch.py: Code to run gridsearch over model params. Provide as input in CLI, the model name(baseline or melody).
	- train.py : Code to train model. Provide as input in CLI, the model name(baseline or melody). The plots for training are also generated after it is complete.
	- test.py : Code to test model. Provide as input in CLI, the model name(baseline or melody). We generate a classification report and also save the results to a csv for analysis.
	

### Training
In CLI, run: `python train.py [model_name]` \(model name is baseline or melody\)

### Testing
In CLI, run: `python test.py [model_name]` \(model name is baseline or melody\)

### Gridsearch
In CLI, run: `python gridsearch.py [model_name]` \(model name is baseline or melody\)

### Contributions
Made by Anmol, Nikunj and Oskar in collaboration with the Digital and Cognitive Musicology Lab at EPFL.