# MMD
This is the implementation of MKGITN described in PACLIC 2021 paper Knowledge Grounded Multimodal Dialog Generation in Task-oriented Settings.
Prerequisites
The recommended way to install the required packages is using pip and the provided requirements.txt file. Create the environment by running the following command:

Mac OS: pip install -r requirements.txt

Linux: Will be released soon.

Download Dataset
Will be released soon

Train and inference
For training, edit config.yml and set is_train: True. Run python train.py. Training result will be output to ./training_output.
For inference, edit config.yml, set is_train: False and test_model_path: 'Your Model Path'. Run python inference.py. Generated responses will be output to ./inference_output.
