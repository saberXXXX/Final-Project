# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_designer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1131, 757)
        MainWindow.setStyleSheet("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.openimage = QtWidgets.QPushButton(self.centralwidget)
        self.openimage.setGeometry(QtCore.QRect(110, 100, 93, 28))
        self.openimage.setObjectName("openimage")
        self.test = QtWidgets.QPushButton(self.centralwidget)
        self.test.setGeometry(QtCore.QRect(100, 430, 93, 28))
        self.test.setObjectName("test")
        self.saveimage = QtWidgets.QPushButton(self.centralwidget)
        self.saveimage.setGeometry(QtCore.QRect(110, 160, 93, 28))
        self.saveimage.setObjectName("saveimage")
        self.result = QtWidgets.QPushButton(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(100, 490, 93, 28))
        self.result.setObjectName("result")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(480, 40, 500, 250))
        self.label.setBaseSize(QtCore.QSize(0, 0))
        self.label.setStyleSheet("background-color: rgb(171, 173, 173);")
        self.label.setObjectName("label")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(480, 370, 500, 250))
        self.label2.setBaseSize(QtCore.QSize(0, 0))
        self.label2.setStyleSheet("background-color: rgb(171, 173, 173);")
        self.label2.setObjectName("label2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1131, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openimage.setText(_translate("MainWindow", "openimage"))
        self.test.setText(_translate("MainWindow", "test"))
        self.saveimage.setText(_translate("MainWindow", "saveimage"))
        self.result.setText(_translate("MainWindow", "result"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label2.setText(_translate("MainWindow", "TextLabel"))