# paper-2018-HAASD
HAASD: A dataset of Household Appliances Abnormal Sound Detection - paper replication data

https://dl.acm.org/citation.cfm?id=3297186

This project uses deep learning to detect the abnormal sound of home appliances such as washing machines, so as to realize the automatic classification of faulty home appliances and normal home appliances. The project contents are as follows:

First, the use of tools
1, programming language: python
2, IDE: pycharm
3. Deep learning framework: Tensorflow
4, matlab

Second, the data set
T10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz four files are test set, test set Tag set, training set, training set tag set
Tmp: This folder contains the data set. After creating the project, please store the folder in the root directory of the disk where the project is located.
The two .m files in this folder are the ones used to process the raw data, one is to generate the training set and its tags, and the other is to generate the test set and its tags, and process the raw data to generate binary files.

Third, the code
Swallowsound: This folder contains the main program of the washing machine abnormal sound detection project, create a new project in pycharm and create a package called swallowsound, put all the files in the folder into the folder where the created package is located. You can run noise_deep_train.py, a trained model, and the output is an accuracy rate.
Swallowsound_input_data.py: This program is used to import training sets, packaged separately into a module for other modules to call, improve code reuse rate
Noise_softmax.py: This program uses softmax as a classifier to easily classify data.
Noise_deep_3cnn.py: This program uses a three-layer convolution kernel convolutional neural network to train data and give prediction results.
Logs: This folder contains the visualization files saved during the training process. Run it in the terminal: tensorboard --logdir=D:\pycharm\swallowsound\swallowsound\logs\ and follow the prompts to see the visualization in the browser. content
Mynet: This folder stores the data stored during training. After the training program is interrupted, it can be used for the next training session.

Fourth, evaluation
Predicted results are accurate up to 85%

If you have any questions or problems with the running program, you can contact WeChat, welcome to communicate and make progress together! Email: s.barry1994@foxmail.com


