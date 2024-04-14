import cv2 as cv
import my_functions as mf
from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
import ctypes
import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog
)

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Megic Eye")
        iconpath = os.path.join((str(os.path.dirname(os.path.abspath(__file__)))), 'eyeicon.png')
        self.setWindowIcon(QtGui.QIcon(iconpath))
        
        #set icon on the taskbar
        QApplication.setWindowIcon(QtGui.QIcon(iconpath))

        #set window Geometry
        self.setMinimumSize(250, 250)
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        width = int(screen.width())
        height = int(screen.height())
        self.setGeometry(int(width/4), int(height/4), int(width/2), int(height/2))

        # Create an outer layout
        outerLayout = QVBoxLayout()
        # Create a form layout for the label and line edit
        topLayout = QFormLayout()
        
        #create widget
        self.text1 = QLineEdit()
        self.text1.mouseDoubleClickEvent = lambda event: self.text1_clicked(self.text1)

        #set look of widget
        self.text1.setMaxLength(80)

        # Add a label and a line edit to the form layout
        topLayout.addRow("File Path:", self.text1)

        #create checkboxes
        self.checkbox1=QCheckBox("face detection")
        self.checkbox2=QCheckBox("object detection")
        self.checkbox3=QCheckBox("edge detection")
        self.checkbox4=QCheckBox("segmentation")
     
        # Add some checkboxes to the layout
        topLayout.addWidget(self.checkbox1)
        topLayout.addWidget(self.checkbox2)
        topLayout.addWidget(self.checkbox3)
        topLayout.addWidget(self.checkbox4)

        #add botoom widgets
        self.text2 = QLineEdit()
        self.text2.mouseDoubleClickEvent = lambda event: self.text2_clicked(self.text2)

        #set look of widget
        self.text2.setMaxLength(80)

        # Add a label and a line edit to the form layout 
        topLayout.addRow("Save Folder:", self.text2)
        self.checkbox5 = QCheckBox("generate text file")
        topLayout.addWidget(self.checkbox5)

        button = QPushButton("Run", self)
        button.clicked.connect(self.button_clicked)
        topLayout.addWidget(button)
        
        # Nest the inner layouts into the outer layout
        outerLayout.addLayout(topLayout)
        
        #create container
        container = QWidget()
        container.setLayout(outerLayout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)

    def button_clicked(self):
        file_path = self.text1.text().strip()
        dir_path = self.text2.text().strip()
        if len(file_path) == 0 or len(dir_path) == 0:
            print("warning")
        elif os.path.exists(file_path) and os.path.exists(dir_path):
            img = mf.img_load(file_path)
            if self.checkbox1.isChecked():
                f_det = mf.face_detection(file_path)
                nf_name = "face_detection_"+str(os.path.split(file_path)[1]) 
                path = os.path.join(dir_path, nf_name)
                cv.imwrite(path, f_det)

            if self.checkbox2.isChecked():
                obj_det = mf.object_detection(img)
                nf_name = "object_detection_"+str(os.path.split(file_path)[1]) 
                path = os.path.join(dir_path, nf_name)
                cv.imwrite(path, obj_det)

            if self.checkbox3.isChecked():
                edge_det = mf.edges_detection(img)
                nf_name = "edges_detection_"+str(os.path.split(file_path)[1]) 
                path = os.path.join(dir_path, nf_name)
                cv.imwrite(path, edge_det)

            if self.checkbox4.isChecked():
                seg = mf.img_segmentation(img)
                nf_name = "segmentation_"+str(os.path.split(file_path)[1]) 
                path = os.path.join(dir_path, nf_name)
                cv.imwrite(path, seg)

            if self.checkbox5.isChecked():
                facenum, obj = mf.img_info(file_path)
                txt_file = str(os.path.split(file_path)[1])[:-3] + "txt"
                path = os.path.join(dir_path, txt_file)
                with open(path, 'w') as file:
                    if facenum > 0 :
                        file.write("Number of detected faces: %d\n" %facenum)
                    else:
                        file.write("No face found.\n")
                    if len(obj) > 0 : 
                        file.write("Found: \n")
                        for i, element in enumerate(obj):
                            text = str(i+1)+"."+str(element)+": "+str(obj[element])+"\n"
                            file.write(text)
                    else: 
                        file.write("No objects found.\n")
           
    def text1_clicked(self, text1):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "", "*.jpg")
        if filename:
            text1.setText(f"{filename}")

    def text2_clicked(self, text2):
        dirpath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirpath:
            text2.setText(f"{dirpath}")

        
        

        

       
        


        

       
        


        

    