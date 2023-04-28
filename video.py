from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5 import uic
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

class videoPlayer:
    def __init__(self):
        self.ui = uic.loadUi('video.ui')  # 加载designer设计的ui程序
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.widget)
        self.ui.pushButton.clicked.connect(self.openVideoFile)
        self.ui.pushButton_2.clicked.connect(self.savevideo)
    # 打开视频文件并播放
    def openVideoFil(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
        self.player.play()

    def savevideo(self):
        self.fname, ftype = QFileDialog.getSaveFileName(self, 'save file', './', "ALL (*.*)")

    def openVideoFile(self):
        self.file, fileType = QFileDialog.getOpenFileName(self, 'open file', './', "ALL (*.*)")






if __name__ == "__main__":
    app = QApplication([])
    myPlayer = videoPlayer()
    myPlayer.ui.show()
    app.exec()
