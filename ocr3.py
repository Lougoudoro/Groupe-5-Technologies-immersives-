# %% [markdown]
# # Import of Libs

# %%
from PIL import Image
import numpy as np
import math
from os import listdir
from os.path import isfile, join, realpath
import json

# %% [markdown]
# ## Definition of constants

# %%
IMGPATH = "sample10.png"
DATASETPATH = "./dataset/trainingData/"
SETS = ["majuscules", "minuscules", "numericals", "specials"]
#SETS = ["t"]
OUTPUTFILE = "./output.json"
GRADE = 200
GRAYSCALE = [0.2989, 0.5870, 0.1140]

# %% [markdown]
# ## Definition of variables

# %%
img = None # original image
chanels = [] # list of chanels
selectedChanel = []
grayScaleChanels = [] # list of grayscaled chanels
firstLineRow = None # defines the 1st line where a black dot is detected
rawLines = dict() # contains rows where black dots are spotted and the column of the black dot. The key is the row and the value is a list of columns
spotedLines = [] # start and ending row for each detected line
rowsLetters = [] # separated letter boundaries [tl, tr, br, bl]
normalizedLetters = [] # normalized letters
binarizedChars = [] # binarized matrix of the characters

# %% [markdown]
# ## Opening the image with Pillow

# %%
img = Image.open(IMGPATH)
display(img)

# %% [markdown]
# ## 2.1 Grayscaling
# ### Formula: Y = 0.2989R+0.5870G+0.1140B

# %% [markdown]
# ### 1. Convertion of the image into a list of matrices each one representing R, G, B and alpha

# %%
chanels = np.array(img)

# %% [markdown]
# ### 2. Function to convert the matrix in to a level of GrayScale

# %%
def grayScale(matrix, scale):
    grayImage = np.zeros(matrix.shape)
    R = np.array(matrix[:, :, 0])
    G = np.array(matrix[:, :, 1])
    B = np.array(matrix[:, :, 2])

    R = (R * scale[0])
    G = (G * scale[1])
    B = (B * scale[2])

    Y = R+G+B

    for i in range(3):
       grayImage[:,:,i] = Y

    return Image.fromarray(np.uint8(grayImage))


# %% [markdown]
# #### Converting the list of matrices in to a grayscale

# %%
grayScaleChanels = grayScale(chanels, [0.2989, 0.5870, 0.1140]) 

# %% [markdown]
# ## 2.2 Feature Extraction

# %%
chanel = np.array(grayScaleChanels)[:, :, 0]

# %% [markdown]
# #### Display of the selected chanel converted into grayscale

# %%
display(Image.fromarray(chanel))

# %% [markdown]
# ##### finds the first line where a black dot is spotted

# %%
def spotFirstLineRow(chanel, grade):
    blackDots = []
    for x in range(len(chanel)):
        for y in range(len(chanel[x])):
            if chanel[x][y] >= grade:
                blackDots.append(x)
    return min(blackDots)

# %%
firstLineRow = spotFirstLineRow(chanel, GRADE)

# %% [markdown]
# ##### removes blank gaps between lines and creates segments

# %%
def remblankLines(chanel, grade, startingRow):
    blackDottedLines = dict()
    for x in range(startingRow, len(chanel)):
        blackDots = []
        for y in range(len(chanel[x])):
            if chanel[x][y] < grade:
                blackDots.append(y)
        blackDottedLines[x] = blackDots
    
    copyOfBlackDottedLines = blackDottedLines.copy()
    for row, cols in blackDottedLines.items():
        if cols==[]:
            del copyOfBlackDottedLines[row]
    return copyOfBlackDottedLines

# %%
rawlinesRows = remblankLines(chanel, GRADE, firstLineRow).keys()
rawlinesRows = list(rawlinesRows)

# %% [markdown]
# ##### segments lines

# %%
def findLinePositions(rawLinesRows):
    linesRows = []
    start = rawLinesRows[0]
    end = start
    for index, line in enumerate(rawlinesRows):
        if index == 0:
            continue
        if rawlinesRows[index-1]+1 != rawlinesRows[index]:
            end = rawlinesRows[index-1]
            linesRows.append((start, end))
            start = rawlinesRows[index]
        if len(rawLinesRows)-1 == index:
            linesRows.append((start, rawlinesRows[index]))
    return linesRows

# %%
spotedLines = findLinePositions(rawlinesRows)

# %% [markdown]
# ##### spot gaps between letters on a single line

# %%
def spotLineEmptyCols(line, chanel, grade):
    cols = []
    for col in range(len(chanel[0])):
        whiteCols = []
        for currentRow in range(line[0], line[1]+1):
            if chanel[currentRow][col] > grade:
                whiteCols.append(col)
        cols.append(whiteCols)
        
    emptyCols = cols.copy()
    for col in cols:
        if len(col) < line[1]-line[0] + 1:
            emptyCols.remove(col)
            
    cols = []
    for listFromCol in emptyCols:
        cols.append(listFromCol[0])
    return cols

# %%
rowsBlankSpaces = []              
for line in spotedLines:
    empty = spotLineEmptyCols(line, chanel, 128)
    rowsBlankSpaces.append((line, empty))

rowsBlankSpaces

# %% [markdown]
# #### spot letters boundaries

# %%
def rowLettersBoundaries(rowBlankSpace):
    letters = [] # col references of the columns
    letterCols = [] # current letter columns
    lettersMatrices = [] # set of letters
    letterMatrix = [] # top left, top right, bottom right and bottom left of the letter
    for col in range(len(chanel[0])):
        if col in rowBlankSpace[1]:
            if letterCols != []:
                letters.append(letterCols)
            letterCols = []
        else:
            letterCols.append(col)

    rows = rowBlankSpace[0]
    for letterCols in letters:
        letterMatrix.append((rows[0], letterCols[0]))
        letterMatrix.append((rows[0], letterCols[len(letterCols)-1]))
        letterMatrix.append((rows[len(rows)-1], letterCols[len(letterCols)-1]))
        letterMatrix.append((rows[len(rows)-1], letterCols[0]))

        lettersMatrices.append(letterMatrix)
        letterMatrix = []
    return lettersMatrices

# %%
for rowBlankSpace in rowsBlankSpaces:
    rowsLetters.append(rowLettersBoundaries(rowBlankSpace))

rowsLetters

# %% [markdown]
# ### Normalization

# %%
def normalizeCharacter(imageArray, top, bottom, left, right, target_size=(15, 15)):
    cropped = np.array(imageArray)[top:bottom, left:right]
    cropped_image = Image.fromarray(cropped)
    normalized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

    return np.array(normalized_image)

# %%

for row in rowsLetters:
    for letter in row:
        normalizedLetters.append(normalizeCharacter(chanel, letter[0][0], letter[2][0], letter[0][1], letter[1][1],))

#saves croped characters
for index, letter in enumerate(normalizedLetters):
    Image.fromarray(letter).save("./test/" + str(index)+".png")

# %% [markdown]
# ### Binarization of the character

# %%
def binarize(character, grade):
    binarized = np.zeros(character.shape)
    for row in range(len(character)):
        for col in range(len(character[row])):
            if character[row][col]<=grade:
                binarized[row][col] = 1
    return binarized

# %%

for index, letter in enumerate(normalizedLetters):
    binarizedChars.append(binarize(letter, 128))

# %% [markdown]
# ## 2.3 Recognition of Pattern

# %% [markdown]
# ### creates track sectors

# %%
def trackSectors(binarizedChar):
    width, height = binarizedChar.shape
    centerY, centerX = width // 2, height // 2

    def pseudoY(centerY, row):
        Y = None
        Y = centerY - row
        if row < centerY:
            Y = - Y
        return Y
    
    def psuedoX(CenterX, col):
        X = None
        X = centerX - col
        if row < centerX:
            X = - X
        return X

    distances = []
    for row in range(len(binarizedChar)):
        Y = pseudoY(centerY, row)
        for col in range(len(binarizedChar[row])):
            X = psuedoX(centerX, col)
            distances.append(np.sqrt((X - centerX)**2 + (Y - centerY)**2))
    rad = max(distances)

    print(rad)

    #size of the tracks
    trackSize = rad / 5

    sectors = []
    for trackNumer in range(5):
        sector = []
        for row in range(len(binarizedChar)):
            Y = pseudoY(centerY, row)
            for col in range(len(binarizedChar[row])):
                X = psuedoX(centerX, col)
                distance = np.sqrt((X - centerX)**2 + (Y - centerY)**2)
                angle = math.atan(Y/X)
                if trackSize * trackNumer < distance and distance <= trackSize * (trackNumer + 1):
                    sector.append(binarizedChar[row][col])
        sectors.append(sector)

    print(sectors)
    return sectors
            

# %%
def track_sectors(binarized_char):
    height, width = binarized_char.shape
    center_y, center_x = height // 2, width // 2

    # Calculate maximum radius (distance from center to farthest pixel)
    max_distance = np.sqrt(center_x**2 + center_y**2)

    # Define track size
    track_size = max_distance / 5

    # Initialize sectors: 5 tracks, each divided into 8 angular sections
    sectors = [[0 for _ in range(8)] for _ in range(5)]

    for row in range(height):
        for col in range(width):
            if binarized_char[row][col] == 1:  # Only consider black pixels
                # Compute distance and angle
                delta_y = row - center_y
                delta_x = col - center_x
                distance = np.sqrt(delta_x**2 + delta_y**2)
                angle = math.atan2(delta_y, delta_x)  # Angle in radians

                # Normalize angle to [0, 2π]
                if angle < 0:
                    angle += 2 * math.pi

                # Determine track index (0-4)
                track_idx = int(distance // track_size)

                # Determine sector index (0-7)
                sector_idx = int((angle / (2 * math.pi)) * 8)

                # Update the count in the corresponding track-sector
                if track_idx < 5:  # Ignore points outside the max radius
                    sectors[track_idx][sector_idx] += 1

    return sectors


# %%
for char in binarizedChars:
    sectors = track_sectors(char)
    print(sectors)
    break

# %% [markdown]
# ## 2.4 Recognition of Output

# %% [markdown]
# #### data set: https://github.com/MinhasKamal/AlphabetRecognizer/tree/master

# %%
def selectChanel(imagePath):
    # opening the image with PILLOW
    img = Image.open(imagePath)
    # chabels extraction
    chanels = np.array(img)
    # converting matrices into grayscale
    grayScaleChanels = grayScale(chanels, GRAYSCALE)
    # selection of a single channel
    selectedChanel = np.array(grayScaleChanels)[:, :, 0]
    return selectedChanel

# %%


# %% [markdown]
# #### Processing of the data set

# %%
def charactersProcessor(img):
    chanels = [] # list of chanels
    selectedChanel = []
    grayScaleChanels = [] # list of grayscaled chanels
    firstLineRow = None # defines the 1st line where a black dot is detected
    rawLines = dict() # contains rows where black dots are spotted and the column of the black dot. The key is the row and the value is a list of columns
    spotedLines = [] # start and ending row for each detected line
    rowsLetters = [] # separated letter boundaries [tl, tr, br, bl]
    normalizedLetters = [] # normalized letters
    binarizedChars = [] # binarized matrix of the characters
    tracks = []
    charTracks = []

    selectedChanel = selectChanel(img)
    # finds the first row
    firstLineRow = spotFirstLineRow(selectedChanel, GRADE)
    # detection of raw lines
    rawlinesRows = remblankLines(selectedChanel, GRADE, firstLineRow).keys()
    rawlinesRows = list(rawlinesRows)
    # finds lines positions
    spotedLines = findLinePositions(rawlinesRows)
    # segment letters of the line
    rowsBlankSpaces = []              
    for line in spotedLines:
        empty = spotLineEmptyCols(line, selectedChanel, 128)
        rowsBlankSpaces.append((line, empty))

    for rowBlankSpace in rowsBlankSpaces:
        rowsLetters.append(rowLettersBoundaries(rowBlankSpace))

    # normalize characters
    for row in rowsLetters:
        for letter in row:
            normalizedLetters.append(normalizeCharacter(chanel, letter[0][0], letter[2][0], letter[0][1], letter[1][1],))

    # binarize characters
    for index, letter in enumerate(normalizedLetters):
        binarizedChars.append(binarize(letter, GRADE))

        # print(np.array(binarizedChars).shape)

    # extract sectors
    for char in binarizedChars:
        charTracks.append(track_sectors(char))
    
    return charTracks

# %%
def char2Track(path):
    # opening the image with PILLOW
    img = Image.open(path)
    # chabels extraction
    chanel = np.array(img)
    # normalize
    normalized = normalizeCharacter(chanel, 0, len(chanel), 0, len(chanel[0]))
    # binarize
    binarizedChar = binarize(normalized, GRADE)
    return track_sectors(binarizedChar)

# %%
chars = dict()

for setPath in SETS:
    path = DATASETPATH+setPath
    path = realpath(path)
    
    for folder in listdir(path):
        newPath = realpath(path+"/"+folder+"/")
        charTracks = []
        for file in listdir(newPath):
            fullPath = join(newPath, file)
            if isfile(fullPath):
                res = char2Track(fullPath)
                charTracks.append(res)
        chars[folder] = charTracks

with open(OUTPUTFILE, "w") as f:
    json.dump(chars, f)



# %% [markdown]
# ### Matrices Similarity comparison

# %%
def matrix_hamming_distance(matrixA, matrixB):
    """
    Calculate the total Hamming distance between two 5x8 matrices.
    """
    if len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0]):
        print(f"{len(matrixA)}--{len(matrixB)}")
        print(f"{len(matrixA[0])}--{len(matrixB[0])}")
        raise ValueError("Matrices must have the same dimensions.")
    
    total_distance = 0
    for rowA, rowB in zip(matrixA, matrixB):
        total_distance += sum(a != b for a, b in zip(rowA, rowB))
    
    return total_distance


# %%
def euclidean_distance(vectorA, vectorB):
    """
    Calculate the Euclidean distance between two vectors.
    """
    vectorA = np.array(vectorA, dtype=object)
    vectorB = np.array(vectorB, dtype=object)
    return sum((a - b) ** 2 for a, b in zip(vectorA, vectorB)) ** 0.5

# %%
with open(OUTPUTFILE, "r") as f:
    dataSet = json.loads(f.read())

    charsTracks = charactersProcessor(IMGPATH)

    result = "-"

    for charTracks in charsTracks:
        charScores = dict()
        for charName, tracks in dataSet.items():
            scores = []
            for track in tracks:
                score = matrix_hamming_distance(charTracks, track)
                scores.append(score)
            charScores[charName] = max(scores)
        print(charScores)
        s = list(charScores.values())
        maxScore = min(s)
        index = s.index(maxScore)
        result = result + list(charScores.keys())[index]
    
    print(result)




# %%
# Load dataset from JSON file
with open(OUTPUTFILE, "r") as f:
    dataSet = json.loads(f.read())

# Process input image to extract character tracks
charsTracks = charactersProcessor(IMGPATH)

result = ""

# Compare each character track against the dataset
for inputTrack in charsTracks:
    charScores = {}

    # Iterate over dataset characters and their tracks
    for charName, datasetTracks in dataSet.items():
        scores = []
        for datasetTrack in datasetTracks:
            # Compute Hamming distance between input and dataset tracks
            #score = matrix_hamming_distance(inputTrack, datasetTrack)
            score = list(euclidean_distance(charTracks, track))
            scores.append(score)
        
        # Use the best score for this character
        if scores:
            charScores[charName] = min(scores)

    # Determine the character with the highest score
    if charScores:
        maxScore = max(charScores.values())
        bestMatch = max(charScores, key=charScores.get)
        result += bestMatch

# Print the resulting matched characters
print("Matched Result:", result)



