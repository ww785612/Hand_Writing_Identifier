import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# confusion pairs: (1,7) (3,5) (6,8) (7,9)
k = 1           # laplace smoothing parameter
trainDataLines = 140000
rowSize = 28
columnSize = 28
numFeatures = 2 # 0 or 1
numClasses = 10 # 0-9, 10 classes of images to classify
prior = 1 / numClasses
totalImageCnt = 5000
numTestImage = 1000


classImageCnt = np.zeros(numClasses)
parameters = np.zeros((numClasses, numFeatures, rowSize, columnSize))
classPixValCnt = np.zeros((numClasses, numFeatures, rowSize, columnSize))
oddRatio_1over7 = np.zeros((rowSize, columnSize))
oddRatio_3over5 = np.zeros((rowSize, columnSize))
oddRatio_6over8 = np.zeros((rowSize, columnSize))
oddRatio_7over9 = np.zeros((rowSize, columnSize))

def genOddRatios():
    for row in range(0,rowSize):
        for column in range(0,columnSize):
            #P(Fij=1 | c1) / P(Fij=1 | c2) = log(P(Fij=1 | c1)) - log(P(Fij=1 | c2))
            oddRatio_1over7[row][column] = parameters[1][1][row][column] - parameters[7][1][row][column]
            oddRatio_3over5[row][column] = parameters[3][1][row][column] - parameters[5][1][row][column]
            oddRatio_6over8[row][column] = parameters[6][1][row][column] - parameters[8][1][row][column]
            oddRatio_7over9[row][column] = parameters[7][1][row][column] - parameters[9][1][row][column]

trainingImageIdx = 0
trainingImageLabel = []
testImageLabel = []

trainingImage = []
trainingImageIdx = 0
rowStart = trainingImageIdx * rowSize
rowEnd = rowStart + rowSize - 1

def getTestLabel():
    testLabelFile = open("testlabels", "r")
    for sLabel in testLabelFile:
        label = int(sLabel)
        testImageLabel.append(label)
    testLabelFile.close()


def getTrainLabel():
    trainLabelFile = open("traininglabels","r")
    for sLabel in trainLabelFile:
        label = int(sLabel)
        classImageCnt[label] += 1
        trainingImageLabel.append(label)
    trainLabelFile.close()

def updatePixCnt(image, imageIdx):
    Class = trainingImageLabel[imageIdx]

    for row in range(0, rowSize):
        for column in range(0, columnSize):
            if(image[row][column]==' '):
                pixelVal = 0
            else:
                pixelVal = 1

            classPixValCnt[Class][pixelVal][row][column] += 1

def genParameter():
    for Class in range(0,numClasses):
        for pixelVal in range(0, numFeatures):
            for row in range(0,rowSize):
                for column in range(0, columnSize):
                    parameters[Class][pixelVal][row][column] = math.log10( (classPixValCnt[Class][pixelVal][row][column]+k) / (classImageCnt[Class] + (k * numFeatures)))

def inference(image):
    likelihood = np.zeros(10)
    for Class in range(0, 10):
        for row in range(0, rowSize):
            for column in range(0, columnSize):
                if (image[row][column] == ' '):
                    pixelVal = 0
                else:
                    pixelVal = 1
                # add up all the probabilities of each pixel being what it is given class to get likelihood P(e|x)
                likelihood[Class] += parameters[Class][pixelVal][row][column]
    matchedClass = np.argmax(likelihood)
    return  matchedClass

def plotOddRatio():
    x = np.array(range(26))
    y = np.array(range(26))
    grid = oddRatio_1over7
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()

def plotLikelihood(Class):
    x = np.array(range(26))
    y = np.array(range(26))
    grid = parameters[Class][1]
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()

#main
getTrainLabel()
trainDataFile = open("trainingimages","r")
for trainFileLineIdx,trainFileLine in enumerate(trainDataFile):
    trainingImage.append(trainFileLine)
    if (trainFileLineIdx >= rowEnd):
        # train here
        updatePixCnt(trainingImage, trainingImageIdx)
        trainingImageIdx += 1
        rowStart = trainingImageIdx * rowSize
        rowEnd = rowStart + rowSize - 1
        trainingImage = []
trainDataFile.close()
genParameter()



# inferencing
numSuccess = 0
testImage = []
testImageIdx = 0
rowStart = testImageIdx * rowSize
rowEnd = rowStart + rowSize - 1

genOddRatios()
getTestLabel()
testDataFile = open("testimages","r")
for testFileLineIdx,testFileLine in enumerate(testDataFile):
    testImage.append(testFileLine)
    if (testFileLineIdx >= rowEnd):
        # inderence here
        inferenceMatch = inference(testImage)
        if(inferenceMatch == testImageLabel[testImageIdx]):
            numSuccess += 1
        testImageIdx += 1
        rowStart = testImageIdx * rowSize
        rowEnd = rowStart + rowSize - 1
        testImage = []
testDataFile.close()

plotLikelihood(1)
plotLikelihood(7)
plotOddRatio()
print(numSuccess/numTestImage)
