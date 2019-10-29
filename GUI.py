import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QPlainTextEdit, QPushButton, QLineEdit, QTextEdit)
from gensim.models.word2vec import Word2Vec

import numpy as np
import os
from inference import do_eval



class MyModel():
    def __init__(self):
        self.model = Word2Vec.load('./Model/ko_en.mdl')

    def getEmbeddingword(self, i):

        if i == '':
            return ''

        e = self.model.wv[i]

        print(e)

        try :
            output = self.model.wv.similar_by_word(e)

            output = np.array(output)
            print(output)
        except KeyError:
            output = "No matched word in dictionary"

        return str(output)


class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.a = 0

    def OnUserInput(self):
        self.a = self.a + 1

        i = self.UserInputCtrl.toPlainText()

        o = str(do_eval(i))

        self.ResultOutput.setPlainText(o)

    def OnClearButtonClicked(self):
        self.UserInputCtrl.setPlainText("")
        self.ResultOutput.setPlainText("")

    def initUI(self):

        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(800, 600)

        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('Input:'), 0, 0)

        self.UserInputCtrl = QPlainTextEdit(self)
        # self.UserInputCtrl.textChanged.connect(self.OnUserInput)
        grid.addWidget(self.UserInputCtrl, 0, 1)

        self.TranslateButton = QPushButton(self)
        self.TranslateButton.setText("Translate")
        self.TranslateButton.clicked.connect(self.OnUserInput)
        grid.addWidget(self.TranslateButton, 1, 1)

        self.ClearButton = QPushButton(self)
        self.ClearButton.setText("Clear")
        self.ClearButton.clicked.connect(self.OnClearButtonClicked)
        grid.addWidget(self.ClearButton, 1, 2)


        grid.addWidget(QLabel('Result:'), 2, 0)

        self.ResultOutput = QPlainTextEdit()
        self.ResultOutput.setReadOnly(True)
        grid.addWidget(self.ResultOutput, 2, 1)

        self.setWindowTitle('')
        self.setGeometry(300, 300, 800, 600)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    base_dir = os.getcwd()
    checkpoint_dir = os.path.join(base_dir, 'kor2eng-gru-gru/2019-10-28-09-33-08-epoch_001/checkpoint.tar')

    ex = MyApp()
    sys.exit(app.exec_())

