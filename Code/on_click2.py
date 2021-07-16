import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.classify import NaiveBayesClassifier
import pickle
import matplotlib.image as mpimg


useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in useless_words}
def MBTI(IntroExtro,IntuitionSensing ,ThinkingFeeling,JudgingPercieiving,input):
    tokenize = build_bag_of_words_features_filtered(input)
    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)

    mbt = ''

    if (ie == 'introvert'):
        mbt += 'I'
    if (ie == 'extrovert'):
        mbt += 'E'
    if (Is == 'Intuition'):
        mbt += 'N'
    if (Is == 'Sensing'):
        mbt += 'S'
    if (tf == 'Thinking'):
        mbt += 'T'
    if (tf == 'Feeling'):
        mbt += 'F'
    if (jp == 'Judging'):
        mbt += 'J'
    if (jp == 'Percieving'):
        mbt += 'P'
    return (mbt)

temp = {'train' : [81.12443979837917,70.14524215640667,80.03456948570128,79.79341109742592], 'test' : [58.20469312585358,54.46262259027357,59.41315234035509,54.40549600629061]}
results = pd.DataFrame.from_dict(temp, orient='index', columns=['Introvert - Extrovert', 'Intuition - Sensing', 'Thinking - Feeling', 'Judging - Percieiving'])
def tellmemyMBTI(IntroExtro,IntuitionSensing ,ThinkingFeeling,JudgingPercieiving,input, name, traasits=[]):
    a = []
    trait1 = pd.DataFrame([0, 0, 0, 0], ['I', 'N', 'T', 'J'], ['count'])
    trait2 = pd.DataFrame([0, 0, 0, 0], ['E', 'S', 'F', 'P'], ['count'])
    for i in input:
        a += [MBTI(IntroExtro,IntuitionSensing ,ThinkingFeeling,JudgingPercieiving,i)]
    for i in a:
        for j in ['I', 'N', 'T', 'J']:
            if (j in i):
                trait1.loc[j] += 1
        for j in ['E', 'S', 'F', 'P']:
            if (j in i):
                trait2.loc[j] += 1
    trait1 = trait1.T
    trait1 = trait1 * 100 / len(input)
    trait2 = trait2.T
    trait2 = trait2 * 100 / len(input)

    # Finding the personality
    YourTrait = ''
    for i, j in zip(trait1, trait2):
        temp = max(trait1[i][0], trait2[j][0])
        if (trait1[i][0] == temp):
            YourTrait += i
        if (trait2[j][0] == temp):
            YourTrait += j
    traasits += [YourTrait]

    # Plotting

    labels = np.array(results.columns)

    intj = trait1.loc['count']
    ind = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, intj, width, color='royalblue')

    esfp = trait2.loc['count']
    rects2 = ax.bar(ind + width, esfp, width, color='seagreen')

    fig.set_size_inches(10, 7)

    ax.set_xlabel('Finding the MBTI Trait', size=18)
    ax.set_ylabel('Trait Percent (%)', size=18)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 105, step=10))
    ax.set_title('Your Personality is ' + YourTrait, size=20)
    plt.grid(True)

    fig.savefig(name + '.png', dpi=200)
    # plt.show()
    return plt,(YourTrait)
f = open('IntroExtro.pickle', 'rb')
IntroExtro = pickle.load(f)
f.close()
f = open('IntuitionSensing.pickle', 'rb')
IntuitionSensing = pickle.load(f)
f.close()
f = open('ThinkingFeeling.pickle', 'rb')
ThinkingFeeling = pickle.load(f)
f.close()
f = open('JudgingPercieiving.pickle', 'rb')
JudgingPercieiving = pickle.load(f)
f.close()




def testing(filename):
    My_writings = open(filename)
    my_writing = My_writings.readlines()
    my_posts = my_writing[0].split('|||')
    #my_posts=filename
    len(my_posts)



    print('Start')
    plt,trait = tellmemyMBTI(IntroExtro, IntuitionSensing, ThinkingFeeling, JudgingPercieiving, my_posts, 'Divy')
    print(trait)
    return plt,trait

class App(QMainWindow):
    
    def test(filename):
        My_writings = QFileDialog.getOpenFileName()
        my_writing = My_writings.readlines()
        my_posts = my_writing[0].split('|||')
        #my_posts=filename
        len(my_posts)



        print('Start')
        plt,trait = tellmemyMBTI(IntroExtro, IntuitionSensing, ThinkingFeeling, JudgingPercieiving, my_posts, 'Divy')
        print(trait)
        return plt,trait

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 textbox - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 940
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)
        #
        self.textboxr = QLineEdit(self)
        self.textboxr.move(20, 80)
        self.textboxr.resize(280, 40)
        #
        self.textboxe = QLineEdit(self)
        self.textboxe.move(20, 140)
        self.textboxe.resize(280, 40)
        # Create a button in the window
        self.button = QPushButton('Predict', self)
        self.button.move(20, 200)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        #
        # Create a button in the window
        self.buttong = QPushButton('Show', self)
        self.buttong.move(20, 230)

        # connect button to function on_click
        self.buttong.clicked.connect(self.on_show)
        self.show()
        ###########
        
         # Create a button in the window
        self.buttonp = QPushButton('Browse', self)
        self.buttonp.move(50, 290)

        # connect button to function on_click
        self.buttonp.clicked.connect(self.test)
        self.show()
        
        


    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        plt,trait=testing(textboxValue)
        if trait=='INTP':
            exp='Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical.'
        if trait=='ISTJ':
            exp='Quiet, serious, earn success by thoroughness and dependability. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized - their work, their home, their life. Value traditions and loyalty.'
        #add all types here with if
        self.textboxr.setText(trait)
        self.textboxe.setText(exp)
    def on_show(self):
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())