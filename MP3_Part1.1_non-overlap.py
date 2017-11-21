import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# confusion pairs: (1,7) (3,5) (6,8) (7,9)
k = 1           # laplace smoothing parameter
trainDataLines = 140000
rowSize = 28
columnSize = 28
patchRowSize = 2
patchColSize = 2
numRowPatch = int(rowSize/patchRowSize)
numColPatch = int(columnSize/patchColSize)
numFeatures = int(math.pow(2, (patchRowSize*patchColSize))) # 0 or 1
numClasses = 10 # 0-9, 10 classes of images to classify
prior = 1 / numClasses
totalImageCnt = 5000
numTestImage = 1000


classImageCnt = np.zeros(numClasses)
parameters = np.zeros((numClasses, numFeatures, rowSize, columnSize))
classfeatureValCnt = np.zeros((numClasses, numFeatures, rowSize, columnSize))


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

def char2Val(character):
    if (character == ' '):
        featureVal = 0
    else:
        featureVal = 1
    return featureVal
# #Patch:
# # 0 1
# # 2 3
# # feature value (for this patch) = formBinary(val(0),val(1),val(2),val(3))
# # val(0) is the most significant bit
# def getPatchFeatureVal(image, patchIdx):
#     startRowPatchGrid = int(int(patchIdx/numColPatch))
#     if(startRowPatchGrid == 0):
#         startColPatchGrid = patchIdx
#     else:
#         startColPatchGrid = int(int(patchIdx % (startRowPatchGrid*numColPatch)))
#     startRow = startRowPatchGrid*patchRowSize
#     startCol = startColPatchGrid*patchColSize
#     assert(startRow+patchRowSize<=rowSize and startCol+patchColSize<=columnSize)
#     featureValue = char2Val(image[startRow][startCol]) * 8 + char2Val(image[startRow][startCol + 1]) * 4 + char2Val(image[startRow+1][startCol]) * 2 + char2Val(image[startRow+1][startCol+1]) * 1
#     return featureValue

#Patch:
# 0 1
# 2 3
# feature value (for this patch) = formBinary(val(0),val(1),val(2),val(3))
# val(0) is the most significant bit
def getPatchFeatureVal(image, startRowPatchGrid, startColPatchGrid):
    startRow = startRowPatchGrid*patchRowSize
    startCol = startColPatchGrid*patchColSize
    assert(startRow+patchRowSize<=rowSize and startCol+patchColSize<=columnSize)
    featureValue = char2Val(image[startRow][startCol]) * 8 + char2Val(image[startRow][startCol + 1]) * 4 + char2Val(image[startRow+1][startCol]) * 2 + char2Val(image[startRow+1][startCol+1]) * 1
    return featureValue

def getTrainLabel():
    trainLabelFile = open("traininglabels","r")
    for sLabel in trainLabelFile:
        label = int(sLabel)
        classImageCnt[label] += 1
        trainingImageLabel.append(label)
    trainLabelFile.close()

def updateFeatureCnt(image, imageIdx):
    Class = trainingImageLabel[imageIdx]
    for row in range(0, numRowPatch):
        for column in range(0, numColPatch):
            featureVal = getPatchFeatureVal(image, row, column)
            classfeatureValCnt[Class][featureVal][row][column] += 1

def genParameter():
    for Class in range(0,numClasses):
        for featureVal in range(0, numFeatures):
            for row in range(0,rowSize):
                for column in range(0, columnSize):
                    parameters[Class][featureVal][row][column] = math.log10( (classfeatureValCnt[Class][featureVal][row][column]+k) / (classImageCnt[Class] + (k * numFeatures)))

def inference(image):
    likelihood = np.zeros(10)
    for Class in range(0, 10):
        for row in range(0, numRowPatch):
            for column in range(0, numColPatch):
                featureVal = getPatchFeatureVal(image, row, column)
                # add up all the probabilities of each feature being what it is given class to get likelihood P(e|x)
                likelihood[Class] += parameters[Class][featureVal][row][column]
    matchedClass = np.argmax(likelihood)
    return  matchedClass


#main
getTrainLabel()
trainDataFile = open("trainingimages","r")
for trainFileLineIdx,trainFileLine in enumerate(trainDataFile):
    trainingImage.append(trainFileLine)
    if (trainFileLineIdx >= rowEnd):
        # train here
        updateFeatureCnt(trainingImage, trainingImageIdx)
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

# plotLikelihood(1)
# plotLikelihood(7)
# plotOddRatio()
print(numSuccess/numTestImage)
