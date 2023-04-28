
import sys
import ui_designer
import predict


import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore




class MasterWindows(QMainWindow, ui_designer.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MasterWindows, self).__init__(parent)
        self.setupUi(self)
        self.openimage.clicked.connect(self.openImage)
        self.saveimage.clicked.connect(self.saveImage)
        self.test.clicked.connect(self.Test)
        self.result.clicked.connect(self.showresult)

    def Test(self):
        name_1 = './models/ACSP(Smooth L1).pth.tea'
        predict.val_single(0.40, name_1)


    def openImage(self):
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量

        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是默认打开的路径，第四个参数是需要的格式

        imgNamepath, imgType = QFileDialog.getOpenFileName(self.centralwidget, "选择图片", "D:\\", "*.jpg;;*.png;;All Files(*)")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QtGui.QPixmap(imgNamepath).scaled(self.label.width(), self.label.height())

        # 在label控件上显示选择的图片
        self.label.setPixmap(img)

    def saveImage(self):
        # 提取Qlabel中的图片
        img = self.label.pixmap().toImage()
        fpath, ftype = QFileDialog.getSaveFileName(self.centralwidget, "保存图片", "e:\\", "*.jpg;;*.png;;All Files(*)")
        img.save(fpath)

    def showresult(self):
        global imgNamepath
        imgNamepath, imgType = QFileDialog.getOpenFileName(self.centralwidget, "选择图片", "D:\\", "*.jpg;;*.png;;All Files(*)")
        img = QtGui.QPixmap(imgNamepath).scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(img)





if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建GUI
    ui = MasterWindows()  # 创建PyQt设计的窗体对象
    ui.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程
