import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Avoid the annoying tf warnings 

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.style.use('ggplot')

def printHeadLine(name:str = '', mainTitle:bool = True, length:int = 80):
    '''Print title in console:
        #### TITLE ####
        ###############

    Parameters:
        name (str): Title displayed
        mainTitle(bool): Add second '#' line if True
        length(int): Length of lines
    '''

    print('')
    length = max(length,len(name))
    if len(name) > 0:
        firstLine = '#'*((length-len(name))//2) + ' ' + name + ' ' + '#'*((length-len(name))//2)
        print(firstLine)
        if mainTitle:
            print('#'*len(firstLine))
    else:
        print('#'*length)
    print('')

def showModelComparison(modelsDict, sizePlot, commonDescription):
    '''Display the evolution of Loss and Accuracy versus epochs given histories of training

    Parameters:
        modelsDict (Dict{ str:Dict{ str:list[float] } }): Dictionary containing the training history of a model referenced by its name
        sizePlot (list[row,column])
        commonDescription (str): Common caracteristics of the models
    
    Return:
        Matplotlib figure: Comparison of given models
    '''


    fig, axes = plt.subplots(sizePlot[0], sizePlot[1])
    plt.text(x=0.5, y=0.94, s='MNIST dataset, Epochs size: ' + str(sizeTrainingSet) + ' (' + str((sizeTrainingSet*100)//sizeTrainingSetInit) + '%)', fontsize=20, ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.88, s= commonDescription, fontsize=17, ha="center", transform=fig.transFigure)


    for i, (nameModel, history) in enumerate(modelsDict.items()):

        try:
            ax = axes.flatten()[i]
        except:
            ax=axes

        for counter, (name, values) in enumerate(history.items()):
            ax.plot(values, label=name, marker='o', color='blue' if counter%2==0 else 'red', linestyle='--' if counter<2 else '-')
            if counter == 3:
                ax.set_title(nameModel + '\nBest accuracy (epochs=' + str(np.argmax(values)+1) + '): ' + str(round(max(values)*100.0,4)) + '%')
        ax.set_xticks([i for i in range(nbrEpochsMax)])
        ax.legend()
    plt.subplots_adjust(top=0.8, wspace=0.2, right=0.95, left=0.05, bottom=0.05, hspace=0.35)
    plt.show()

def showMispredictions(model):
    '''Show firsts mispredicted images by the given model.

    Parameters:
        modelsDict (tf.keras.models): Trained model to test.
    
    Return:
        Matplotlib figure: Mispredicted images
    '''

    printHeadLine('MODEL TESTING')
    lossANN, accANN = model.evaluate(xTest, yTest)
    predictions = model.predict(xTest)

    wrongPredictionsIndex = []

    for index in range(len(predictions)):
        if not np.argmax(predictions[index]) == yTest[index]:
            wrongPredictionsIndex.append(index)


    hVisu = 4
    wVisu = 4
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Loss: ' + str(lossANN) + '  -  Accuracy:' + str(accANN), fontsize=20)
    outer = gridspec.GridSpec(hVisu, wVisu, wspace=0.2, hspace=0.4)

    for index in range(wVisu*hVisu):
        if index < len(wrongPredictionsIndex):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[index], wspace=0.2, hspace=0.05)

            ax1 = plt.Subplot(fig, inner[0])
            ax1.imshow(xTest[wrongPredictionsIndex[index]], cmap="gray_r", vmin=0.0, vmax=1.0)
            ax1.set_title("Labeled " + str(yTest[wrongPredictionsIndex[index]]))

            ax2 = plt.Subplot(fig, inner[1])
            ax2.bar([i for i in range(10)], predictions[wrongPredictionsIndex[index]])
            ax2.set_xticks([i for i in range(10)])
            ax2.set_title('Predicted:' + str(np.argmax(predictions[wrongPredictionsIndex[index]])))

            fig.add_subplot(ax1)
            ax1.set_aspect('equal')
            fig.add_subplot(ax2)
            ax1.set_aspect('equal')
    plt.show()

## DATA LOADING ##
##################

nbrEpochsMax=15
sizeTrainingSet = 3000 #12000
batchSize = 100
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data() #Load MNIST dataset
sizeTrainingSetInit = len(xTrain)
sizeTrainingSet = min(sizeTrainingSet,sizeTrainingSetInit)
xTrain, yTrain = xTrain[:sizeTrainingSet], yTrain[:sizeTrainingSet]
xTrain, xTest = xTrain / 255.0, xTest / 255.0 #Normalize (or scale) pixel values to [0,1]
yTrain_oneHot = tf.keras.utils.to_categorical(yTrain) # 5 -> [0,0,0,0,0,1,0,0,0,0]
yTest_oneHot = tf.keras.utils.to_categorical(yTest)

img_rows = 28
img_cols = 28
if tf.keras.backend.image_data_format() == 'channels_first':
    xTrain = xTrain.reshape(xTrain.shape[0], 1, img_rows, img_cols)
    xTest = xTest.reshape(xTest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    xTrain = xTrain.reshape(xTrain.shape[0], img_rows, img_cols, 1)
    xTest = xTest.reshape(xTest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


printHeadLine('MNIST dataset')
print('Training size: ' + str(len(xTrain)) + ' out of ' + str(sizeTrainingSetInit))
print('Testing size: ' + str(len(xTest)))
print(tf.keras.backend.image_data_format())

## MODELS DEFINITION ##
#######################

def Comparison_ANN_ActivationNeuronalSize():
    modelsDict = {}
    
    printHeadLine('MODELS TRAINING')

    for nbrNeuron in [64, 512]:
        ## Classic ANN - Linear
        nameModel = 'Linear activation - {} neurons per layers'.format(nbrNeuron)
        ANNsigmoid_model = tf.keras.models.Sequential()
        ANNsigmoid_model.add(tf.keras.layers.Flatten()) #Flattening layer
        ANNsigmoid_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.linear))
        ANNsigmoid_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.linear))
        ANNsigmoid_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
        ANNsigmoid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        printHeadLine(nameModel.upper(), mainTitle=False)
        modelsDict[nameModel] = ANNsigmoid_model.fit(xTrain, yTrain, epochs=nbrEpochsMax, validation_data=(xTest, yTest)).history
        ANNsigmoid_model.summary()

        ## Classic ANN - Relu
        nameModel = 'Relu activation - {} neurons per layers'.format(nbrNeuron)
        ANNrelu_model = tf.keras.models.Sequential()
        ANNrelu_model.add(tf.keras.layers.Flatten()) #Flattening layer
        ANNrelu_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.relu))
        ANNrelu_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.relu))
        ANNrelu_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
        ANNrelu_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        printHeadLine(nameModel.upper(), mainTitle=False)
        modelsDict[nameModel] = ANNrelu_model.fit(xTrain, yTrain, epochs=nbrEpochsMax, validation_data=(xTest, yTest)).history
        ANNrelu_model.summary()

        ## Classical ANN - Sigmoid
        nameModel = 'Sigmoid activation - {} neurons per layers'.format(nbrNeuron)
        ANNsigmoid_model = tf.keras.models.Sequential()
        ANNsigmoid_model.add(tf.keras.layers.Flatten()) #Flattening layer
        ANNsigmoid_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.sigmoid))
        ANNsigmoid_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.sigmoid))
        ANNsigmoid_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
        ANNsigmoid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        printHeadLine(nameModel.upper(), mainTitle=False)
        modelsDict[nameModel] = ANNsigmoid_model.fit(xTrain, yTrain, epochs=nbrEpochsMax, validation_data=(xTest, yTest)).history
        ANNsigmoid_model.summary()

    return modelsDict

def Comparison_ANN_Optimizer():
    modelsDict = {}
    nbrNeuron = 128
    printHeadLine('MODELS TRAINING')

    for optType in ['Adam', 'Gradient descent']:
        for learningRate in [0.001, 0.01, 0.1]:

            nameModel='{} optimizer - {} learning rate'.format(optType, learningRate)
            printHeadLine(nameModel.upper(), mainTitle=False)

            if optType == 'Adam':
                opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
            if optType == 'Gradient descent':
                opt = tf.keras.optimizers.SGD(learning_rate=learningRate)
            
            ANN_model = tf.keras.models.Sequential()
            ANN_model.add(tf.keras.layers.Flatten()) #Flattening layer
            ANN_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.relu))
            ANN_model.add(tf.keras.layers.Dense(nbrNeuron, activation=tf.keras.activations.relu))
            ANN_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
            ANN_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            modelsDict[nameModel] = ANN_model.fit(xTrain, yTrain, epochs=nbrEpochsMax, validation_data=(xTest, yTest)).history
            ANN_model.summary()

    return modelsDict

def Comparison_CNNvANN():
    modelsDict = {}
    printHeadLine('MODELS TRAINING')

    ## CNN - Beefed up version of Yann LeCun's LeNet-5 (cf. Gradient-based laerning applied to document recognition - p.7)
    nameModel='CNN: Conv2D(32) - Conv2D(64) - MaxPooling2D(2x2) - Dense(128)'
    printHeadLine(nameModel.upper(), mainTitle=False)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(xTrain, yTrain_oneHot, epochs=nbrEpochsMax, validation_data=(xTest, yTest_oneHot)).history
    nameModel='CNN ({} params): Conv2D(32) - Conv2D(64) - MaxPooling2D(2x2) - Dense(128)'.format(str(model.count_params()))
    modelsDict[nameModel] = hist
    model.summary()

    ## ANN
    nameModel='ANN: Dense(128) - Dense(256) - Dense(128)'
    printHeadLine(nameModel.upper(), mainTitle=False)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) #Flattening layer
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(xTrain, yTrain, epochs=nbrEpochsMax, validation_data=(xTest, yTest)).history
    nameModel='ANN ({} params): Dense(128) - Dense(256) - Dense(128)'.format(str(model.count_params()))
    modelsDict[nameModel] = hist
    model.summary()

    return modelsDict

## RESULTS ##
#############

IdActivation, IdOptimizer, IdCnnAnn = range(3)
SELECTOR = IdCnnAnn

if __name__ == "__main__":
    if SELECTOR == IdActivation:
        showModelComparison(Comparison_ANN_ActivationNeuronalSize(), sizePlot=[2,3], commonDescription='2 dense hidden layers - Adam optimizer - Cross entropy loss - Softmax output activation')
    if SELECTOR == IdOptimizer:
        showModelComparison(Comparison_ANN_Optimizer(), sizePlot=[2,3], commonDescription='2 dense hidden layers - Linear rectified activation - 128 Neuron per layer - Cross entropy loss - Softmax output activation')
    if SELECTOR == IdCnnAnn:
        showModelComparison(Comparison_CNNvANN(), sizePlot=[1,2], commonDescription='Adam optimizer - Cross entropy loss - Softmax output activation')