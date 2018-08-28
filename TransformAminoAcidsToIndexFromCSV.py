from AminoAcidDict import *
import pandas as pd
import os

class AminoAcidsCSV:
    def __init__(self, filePath, fileName):
        self.filePath = filePath
        self.fileName = fileName
        self.fullFileName = os.path.join(filePath, fileName)
        self.data = pd.read_csv(self.fullFileName)
        self.indexDF = None

    def checkFormat(self):
        ## the datasets should include only two sequences,
        ## one is the starting sequence and the other is the ending sequences
        if len(self.data.columns.values)!=2:
            raise ValueError("The number of sequences is not two")

    def getColNames(self):
        return self.data.columns.values

    def getStartingSeq(self):
        colNames = self.getColNames()
        return self.data[colNames[0]]

    def getEndingSeq(self):
        colNames = self.getColNames()
        return self.data[colNames[1]]

    def convertSeqToIndex(self, seqVector):
        return aminoAcidsVecToIndex(seqVector)

    def convertStartingSeqToIndex(self):
        seqVector = self.getStartingSeq()
        return self.convertSeqToIndex(seqVector)

    def convertEndingSeqToIndex(self):
        seqVector = self.getEndingSeq()
        return self.convertSeqToIndex(seqVector)


    def combineIndexSeqIntoPD(self):
        startingSeq = self.convertStartingSeqToIndex()
        endingSeq = self.convertEndingSeqToIndex()
        ## combine the two lists as two columns into a data frame and assign column names for the data frame
        df = pd.DataFrame( {self.getColNames()[0]:startingSeq, self.getColNames()[1]: endingSeq})
        self.indexDF = df
        return df




# aminoAcidCSV = AminoAcidsCSV("/Users/crystal/Dropbox/rejfreePy_main/ReadDataset", "Seq1_29Seq2_289.csv")
# sequences = aminoAcidCSV.data
# colNames = sequences.columns.values
# ## ToDo: map each column to a number and add them column by column to a data frame
# ## this is how we extract one column sequences['nonGapSeq1']
#
#
# ## get all the column names of sequences
# print(sequences.columns.values)
# startingSeq = aminoAcidCSV.getStartingSeq()
# endingSeq = aminoAcidCSV.convertStartingSeqToIndex()
# x = pd.DataFrame(list(zip(startingSeq, endingSeq)))
# print(x.head(6))
# print(aminoAcidCSV.combineIndexSeqIntoPD().head(6))