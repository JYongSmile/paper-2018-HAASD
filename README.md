# paper-2018-HAASD
HAASD: A dataset of Household Appliances Abnormal Sound Detection - paper replication data


1053/5000
Cǐ xiàngmù shì shǐyòng shēndù xuéxí duì xǐyījī děng jiādiàn yì yīn jīng xíng jiǎncè, cóng'ér shíxiàn zìdòng duì gùzhàng jiādiàn yǔ zhèngcháng jiādiàn jīng háng fēnlèi de mùdì. Xiàngmù nèiróng rúxià:

Yī, shǐyòng gōngjù
1, chéngxù yǔyán:Python 
2,IDE:Pycharm
3, shēndù xuéxí kuàngjià:Tensorflow
4,matlab

èr, shùjù jí
t10k-images-idx3-ubyte.Gz,t10k-labels-idx1-ubyte.Gz,train-images-idx3-ubyte.Gz,train-labels-idx1-ubyte.Gz sì gè wénjiàn fēnbié shì cèshì jí, cèshì jí de biāoqiān jí, xùnliàn jí, xùnliàn jí de biāoqiān jí
tmp: Gāi wénjiàn jiā lǐ cúnfàng de shì shùjù jí, zài chuàngjiàn xiàngmù hòu, qǐng jiāng gāi wénjiàn jiā cúnfàng rù xiàngmù suǒzài cípán de gēn mùlù zhōng
cǐ wénjiàn jiā zhōng de liǎng gè.M wénjiàn shì chǔlǐ yuánshǐ shùjù suǒyòng chéngxù, yīgè shì shēngchéngxùnliàn jí jí qí biāoqiān, lìng yīgè shì shēngchéng cèshì jí jí qí biāoqiān, jiāng yuánshǐ shùjù chǔlǐ shēngchéng èrjìnzhì wénjiàn

sān, dàimǎ
swallowsound: Gāi wénjiàn jiā lǐ cúnfàng xǐyījī yì yīn jiǎncè xiàngmù dì zhǔyào chéngxù, zài pycharm zhōng chuàngjiàn yīgè xīn xiàngmù bìngqiě chuàngjiàn yīgè míng wèi swallowsound de bāo, bǎ gāi wénjiàn jiā zhōng suǒyǒu wénjiàn fàng rù chuàngjiàn de zhège bāo suǒzài wénjiàn jiā zhōng jí kě yùnxíng noise_deep_train.Py zhège yǐjīng xùnliàn hǎo de móxíng, shūchū jiéguǒ shì yīgè zhǔnquè lǜ
swallowsound_input_data.Py: Gāi chéngxù shì yòng lái dǎorù xùnliàn jí, dāndú fēngzhuāng chéng yīgè mókuài kě gōng qítā mókuài diàoyòng, tígāo dàimǎ de chóngfù shǐyòng lǜ
noise_softmax.Py: Gāi chéngxù shì shǐyòng softmax zuòwéi fēnlèi qì duì shùjù jīng xíngle jiǎndān fēnlèi
noise_deep_3cnn.Py: Gāi chéngxù shì shǐyòng sān céng juàn jī hé juàn jī shénjīng wǎngluò duì shùjù jīng xíng xùnliàn bìng gěi chū yùcè jiéguǒ
logs: Gāi wénjiàn jiā lǐ cúnfàngle xùnliàn guòchéng zhōng bǎocún de kěshìhuà wénjiàn, zài terminal zhōng yùnxíng:Tensorboard --logdir=D:\Pycharm\swallowsound\swallowsound\logs\ hòu kě ànzhào tíshì jí kě zài liúlǎn qì zhòng kàn dào kěshìhuà nèiróng
mynet: Gāi wénjiàn jiā zhōng cúnfàng de shì xùnliàn shí cúnfàng de shùjù, zài xùnliàn chéngxù zhōngduàn hòu kě gōng xià cì jiēzhe běn cì xùnliàn jiéguǒ jìxù xùnliàn

sì, pínggū
yùcè jiéguǒ de zhǔnquè lǜ zuìgāo dádào 85%

rúguǒ yǒu shé me yíwèn huòzhě yùnxíng chéng xù chūxiàn shénme wèntí kěyǐ wēixìn liánxì, huānyíng jiāoliú gōutōng, gòngtóng jìnbù!	Yóuxiāng:S.Barry1994@foxmail.Com
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
