"""Image Style Transfer by using CNN
code Written in python, Gui made with PyQt5"""
from PIL import Image
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFile, QTextStream
import threading
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5 import QtWidgets
import ctypes

#don't delete using python files with image and css source
import css
import os

#global variables for using in the entire code.
global content_path
global style_path
global output_image
global flag_content_image
flag_content_image=0
global flag_style_image
flag_style_image=0
global flag_finish_generate
flag_finish_generate=0
global count
count=0
global number_of_iterations
number_of_iterations = 0

"""main_window_gui is the main class of the GUI,
all GUI parameters and code functions defined here."""
class main_window_gui(QWidget):
    def __init__(self, parent=None):
        super(main_window_gui, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h
        self.initUI()

    #Exit all the program
    def closeEvent(self, QCloseEvent):
            os._exit(0)

    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":css/Icons/logo.png"))
        self.setGeometry(0, 0, self.width, self.height-60)

        #Creating main frame,it is a container for the all framers. parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # the Icons sub frame
        self.iconsub_frame = QtWidgets.QFrame(self.main_frame)
        self.iconsub_frame.setFixedHeight(80)
        self.main_layout.addWidget(self.iconsub_frame)
        self.iconsub_layout = QtWidgets.QHBoxLayout(self.iconsub_frame)
        self.iconsub_layout.setAlignment(Qt.AlignLeft)

        # help button
        help_btn = QtWidgets.QPushButton("", self)
        help_btn.setObjectName("TransparentButtons")
        help_btn.setStyleSheet("QPushButton {background: url(:css/Icons/help.png) no-repeat transparent;}")
        help_btn.setToolTip('Show help pdf.')
        help_btn.setFixedWidth(68)
        help_btn.setFixedHeight(68)
        help_btn.clicked.connect(self.show_help_pdf)
        self.iconsub_layout.addWidget(help_btn)

        # the Icon sub frame
        self.logo_sub_frame = QtWidgets.QFrame(self.main_frame)
        self.logo_sub_frame.setFixedWidth(self.width)
        self.main_layout.addWidget(self.logo_sub_frame)
        self.logosub_layout = QtWidgets.QHBoxLayout(self.logo_sub_frame)
        self.logosub_layout.setAlignment(Qt.AlignCenter)

        # Setting up the logo
        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(":css/Icons/logo.png")
        pixmap = pixmap.scaled(260, 260)
        logo.setPixmap(pixmap)
        self.logosub_layout.addWidget(logo)
        logo.setAlignment(Qt.AlignCenter)

        # The Button sub frame
        self.button_sub_frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.button_sub_frame)
        self.button_sub_layout = QtWidgets.QHBoxLayout(self.button_sub_frame)
        self.button_sub_frame.setFixedWidth(self.width)
        self.button_sub_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        # start to create style button
        StartCreateNewBtn = QtWidgets.QPushButton("Style your image", self)
        StartCreateNewBtn.setObjectName("MainGuiButtons")
        StartCreateNewBtn.setToolTip('Start image style process.')
        StartCreateNewBtn.clicked.connect(self.openTransferImageGui)
        self.button_sub_layout.addWidget(StartCreateNewBtn)

        # credit text label on main layout
        creditsLbl = QtWidgets.QLabel('Created By Koral Zakai & May Steinfeld, '
                                      'Supervisor: Zeev Vladimir Volkovich, '
                                      '03/06/2019')
        creditsLbl.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(creditsLbl)

        # show the window on full screen
        self.showMaximized()

    #if clicked on "style your image"
    def openTransferImageGui(self):
        transferImage = TransferImageGui(self)
        transferImage.show()
        self.main_frame.setVisible(False)

    #help- open web browser
    def show_help_pdf(self):
        import os
        import webbrowser
        #filename = 'Help.pdf'
        filename = 'https://drive.google.com/file/d/15odkIk28GARWVzHkBz2gUJ4T3r7X2YZs/view'
        try:
            #os.startfile(filename)
            webbrowser.open(filename)  #Go to help file in googleDrive
        except:
            return

# The main Gui, include uploading 2 images and generate Btn .
class TransferImageGui(QWidget):
    def __init__(self, parent=None):
        super(TransferImageGui, self).__init__(parent)

        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h
        self.initUI2()
        self.t = None

    def initUI2(self):
        global flag_content_image
        flag_content_image = 0
        global flag_style_image
        flag_style_image = 0
        global flag_finish_generate
        flag_finish_generate = 0

        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":css/Icons/logo.png"))
        self.setGeometry(0, 0, self.width, self.height - 60)

        # Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        # the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # home and help buttons
        # the Icons sub frame
        self.iconsub_frame = QtWidgets.QFrame(self.main_frame)
        self.iconsub_frame.setFixedHeight(80)
        self.main_layout.addWidget(self.iconsub_frame)
        self.iconsub_layout = QtWidgets.QHBoxLayout(self.iconsub_frame)
        self.iconsub_layout.setAlignment(Qt.AlignLeft)

        # help button
        help_btn = QtWidgets.QPushButton("", self)
        help_btn.setObjectName("TransparentButtons")
        help_btn.setStyleSheet("QPushButton {background: url(:css/Icons/help.png) no-repeat transparent;}")
        help_btn.setToolTip('Show help pdf.')
        help_btn.setFixedWidth(68)
        help_btn.setFixedHeight(68)
        help_btn.clicked.connect(main_window_gui.show_help_pdf)
        self.iconsub_layout.addWidget(help_btn)

        # home button
        home_btn = QtWidgets.QPushButton("", self)
        home_btn.setObjectName("TransparentButtons")
        home_btn.setStyleSheet("QPushButton {background: url(:css/Icons/home.png) no-repeat transparent;} ")
        home_btn.setFixedWidth(68)
        home_btn.setFixedHeight(68)
        home_btn.setToolTip('Return home screen.')
        home_btn.clicked.connect(self.show_home)
        self.iconsub_layout.addWidget(home_btn)

        self.buttonsSub_Frame = QtWidgets.QFrame(self.main_frame)
        self.buttonsSub_Frame.setFixedWidth(self.width)
        self.buttonsSub_Frame.setFixedHeight(100)
        self.main_layout.addWidget(self.buttonsSub_Frame)
        self.buttonsSub_Layout = QtWidgets.QHBoxLayout(self.buttonsSub_Frame)
        self.buttonsSub_Layout.setAlignment(Qt.AlignCenter|Qt.AlignTop)

        QtCore.QMetaObject.connectSlotsByName(main)

        # upload content button
        contentBtn = QtWidgets.QPushButton("Upload content image", self)
        contentBtn.setObjectName("MainGuiButtons")
        contentBtn.setToolTip('Upload content image.')
        contentBtn.clicked.connect(self.set_content_image)
        self.buttonsSub_Layout.addWidget(contentBtn)

        # upload style
        StyleBtn = QtWidgets.QPushButton("Upload style image", self)
        StyleBtn.setObjectName("MainGuiButtons")
        StyleBtn.setToolTip('Upload style image.')
        StyleBtn.clicked.connect(self.set_style_image)
        self.buttonsSub_Layout.addWidget(StyleBtn)

        #framer for the uploaded content and style images
        self.photosframe = QtWidgets.QFrame(self.main_frame)
        self.photosframe.setFixedWidth(self.width)
        self.main_layout.addWidget(self.photosframe)
        self.photosSub_Layout = QtWidgets.QHBoxLayout(self.photosframe)
        self.photosSub_Layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        #lable for  the content image
        self.contentLabel = QtWidgets.QLabel('', self)
        pixmap = QPixmap(":css/Icons/imageNeedUpload.png")
        pixmap = pixmap.scaled(256, 256)
        self.contentLabel.setPixmap(pixmap)
        self.photosSub_Layout.addWidget(self.contentLabel)
        self.contentLabel.setAlignment(Qt.AlignCenter)

        #lable for the style image
        self.styleLabel = QtWidgets.QLabel('', self)
        self.styleLabel.setPixmap(pixmap)
        self.photosSub_Layout.addWidget(self.styleLabel)
        self.styleLabel.setAlignment(Qt.AlignCenter)

        self.details_Frame = QtWidgets.QFrame(self.main_frame)
        self.details_Frame.setFixedWidth(self.width)
        self.details_Frame.setFixedHeight(60)
        self.main_layout.addWidget(self.details_Frame)
        self.details_Layout = QtWidgets.QHBoxLayout(self.details_Frame)
        self.details_Layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        iterText = QtWidgets.QLabel('Image quality:')
        self.details_Layout.addWidget(iterText)
        self.iterationbox = QtWidgets.QComboBox(self.main_frame)
        self.iterationbox.addItem("Low")
        self.iterationbox.addItem("Medium")
        self.iterationbox.addItem("High")
        self.details_Layout.addWidget(self.iterationbox)

        resText = QtWidgets.QLabel('    Image resolution:')
        self.details_Layout.addWidget(resText)
        self.resolutionbox = QtWidgets.QComboBox(self.main_frame)
        self.resolutionbox.addItem("Small- 256 px")
        self.resolutionbox.addItem("Medium- 512 px")
        self.resolutionbox.addItem("Large- 1024 px")
        self.details_Layout.addWidget(self.resolutionbox)

        modelText = QtWidgets.QLabel('  Choose model:')
        self.details_Layout.addWidget(modelText)
        self.modelBox = QtWidgets.QComboBox(self.main_frame)
        self.modelBox.addItem("Vgg16")
        self.modelBox.addItem("Vgg19")
        #self.modelBox.addItem("ResNet")
        self.details_Layout.addWidget(self.modelBox)

        self.generateBtnSub_Frame = QtWidgets.QFrame(self.main_frame)
        self.generateBtnSub_Frame.setFixedWidth(self.width)
        self.main_layout.addWidget(self.generateBtnSub_Frame)
        self.generateBtnSub_Layout = QtWidgets.QHBoxLayout(self.generateBtnSub_Frame)
        self.generateBtnSub_Layout.setAlignment(Qt.AlignCenter)

        self.generateBtn = QtWidgets.QPushButton("Generate", self)
        self.generateBtn.setToolTip('Generate image.')
        self.generateBtn.setObjectName("MainGuiButtons")
        self.generateBtn.clicked.connect(self.start_thread)
        self.generateBtnSub_Layout.addWidget(self.generateBtn)
        self.generateBtn.setEnabled(True)

        # show the window on full screen
        self.showMaximized()

    """start_thread start the thread that running the Main functions in the program- StyleMakerFunc."""
    def start_thread(self):
        if flag_style_image == 1 and flag_content_image == 1:
            outputWindow = output_imageGui(self)
            outputWindow.getComboBoxValues(self.iterationbox.currentText(), self.resolutionbox.currentText() , self.modelBox.currentText())
            self.t = threading.Thread(target=outputWindow.generate)
            #start the thread
            self.t.start()
            outputWindow.show()
            self.main_frame.setVisible(False)
        else:
            #show error msg
            QMessageBox.critical(self, "Error", "You must upload content and style images first.")

    # Opens home window
    def show_home(self):
        """
        close current window and return to home page
        """
        home = main_window_gui(self)
        home.show()
        self.main_frame.setVisible(False)

    """set_content_image function choosing the content image form dialog file box and ."""
    def set_content_image(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global content_path
            content_path = fileName[0]
            global flag_content_image
            flag_content_image = 1
            try:
                img = Image.open(content_path)  # open the image file
                img.verify()  # verify that it is, in fact an image
                pixmap = QtGui.QPixmap(fileName[0])
                pixmap = pixmap.scaled(256, 256)
                self.contentLabel.setPixmap(pixmap)
            except (IOError, SyntaxError) as e:
                flag_content_image = 0
                pixmap = QPixmap(":css/Icons/imageNeedUpload.png")
                pixmap = pixmap.scaled(256, 256)
                self.contentLabel.setPixmap(pixmap)
                QMessageBox.critical(self, "Error", "Image is corrupted, please upload a good image." )

    """set_style_image function choosing the content image form dialog file box and ."""
    def set_style_image(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global style_path
            style_path = fileName[0]
            global flag_style_image
            flag_style_image = 1
            try:
                img = Image.open(style_path)  # open the image file
                img.verify()  # verify that it is, in fact an image
                pixmap = QtGui.QPixmap(fileName[0])
                pixmap = pixmap.scaled(256, 256)
                self.styleLabel.setPixmap(pixmap)
            except (IOError, SyntaxError) as e:
                flag_style_image = 0
                pixmap = QPixmap(":css/Icons/imageNeedUpload.png")
                pixmap = pixmap.scaled(256, 256)
                self.styleLabel.setPixmap(pixmap)
                QMessageBox.critical(self, "Error", "Image is corrupted , please upload a good image." )

# Show generate the images and show the output result
class output_imageGui(QWidget):
    def __init__(self , parent=None):
        super(output_imageGui, self).__init__(parent)
        self.show
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h
        self.initUI()

    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":css/Icons/logo.png"))
        self.setGeometry(0, 0, self.width, self.height - 60)

        # Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # the Icons sub frame
        self.iconsub_frame = QtWidgets.QFrame(self.main_frame)
        self.iconsub_frame.setFixedHeight(80)
        self.main_layout.addWidget(self.iconsub_frame)
        self.iconsub_layout = QtWidgets.QHBoxLayout(self.iconsub_frame)
        self.iconsub_layout.setAlignment(Qt.AlignLeft)

        # help button
        help_btn = QtWidgets.QPushButton("", self)
        help_btn.setObjectName("TransparentButtons")
        help_btn.setStyleSheet("QPushButton {background: url(:css/Icons/help.png) no-repeat transparent;}")
        help_btn.setToolTip('Show help pdf.')
        help_btn.setFixedWidth(68)
        help_btn.setFixedHeight(68)
        help_btn.clicked.connect(main_window_gui.show_help_pdf)
        self.iconsub_layout.addWidget(help_btn)

        # home button
        self.home_btn = QtWidgets.QPushButton("", self)
        self.home_btn.setObjectName("TransparentButtons")
        self.home_btn.setStyleSheet("QPushButton {background: url(:css/Icons/home.png) no-repeat transparent;} ")
        self.home_btn.setFixedWidth(68)
        self.home_btn.setFixedHeight(68)
        self.home_btn.clicked.connect(self.show_home)
        self.iconsub_layout.addWidget(self.home_btn)
        self.home_btn.setToolTip('Return home screen.')

        # The output image sub frame
        self.output_sub_frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.output_sub_frame)
        self.output_sub_layout = QtWidgets.QVBoxLayout(self.output_sub_frame)
        self.output_sub_frame.setFixedWidth(self.width)
        self.output_sub_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        self.output_frame = QtWidgets.QLabel(self.main_frame)
        self.output_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.output_frame.setText("")
        pixmap = QPixmap(":css/Icons/gift.png")
        pixmap = pixmap.scaled(256, 256)
        self.output_frame.setPixmap(pixmap)
        self.output_frame.setScaledContents(True)
        self.output_frame.setObjectName("output_frame")
        self.output_frame.setAlignment(Qt.AlignCenter)
        self.output_sub_layout.addWidget(self.output_frame)

        # The progressBar sub frame
        self.progressBar_sub_frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.progressBar_sub_frame)
        self.progressBar_sub_frame.setFixedWidth(self.width)
        self.progressBar_sub_layout = QtWidgets.QHBoxLayout(self.progressBar_sub_frame)
        self.progressBar_sub_layout.setAlignment(Qt.AlignCenter)

        self.progressBar = QtWidgets.QProgressBar(self.main_frame)
        self.progressBar.setFixedWidth(self.width/3)
        self.progressBar.setProperty("value",0)
        self.progressBar.setMaximum(100)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.progressBar_sub_layout.addWidget(self.progressBar)

        # The Button save sub frame
        self.button_sub_frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.button_sub_frame)
        self.button_sub_layout = QtWidgets.QVBoxLayout(self.button_sub_frame)
        self.button_sub_frame.setFixedWidth(self.width)
        self.button_sub_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        # Save button for the output image
        self.save_button = QtWidgets.QPushButton("Save your image", self)
        self.save_button.setObjectName("MainGuiButtons")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setToolTip('Save ouput image.')
        self.button_sub_layout.addWidget(self.save_button)

        # show the window
        self.showMaximized()

    #get the selected values from the comboBox
    def getComboBoxValues(self, iteration_string, resolution_string, model_string):
        self.iteration_string = iteration_string
        self.resolution_string  = resolution_string
        self.model_string = model_string

    """save_image function control the saving of the output image."""
    def save_image(self):
        global output_image
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Select Image", "",
                                                            "Image Files (*.jpg *.png *.jpeg *.bmp)")
        if (file_name):
            output_image.save(file_name)

    # Opens home window
    def show_home(self):
        """
        close current window and return to home page
        """
        global flag_finish_generate
        if flag_finish_generate == 0:
            QMessageBox.critical(self, "Error", "You can not return home while process is running.")
        else:
            home = main_window_gui(self)
            home.show()
            self.main_frame.setVisible(False)

    """on_count_changed function updating the progressBar."""
    def on_count_changed(self, value):
        self.progressBar.setValue(value)

    """generate function is start when the generate button pushed. it start the main algorithm."""
    def generate(self):
        global output_image
        self.save_button.hide()

        #number_of_iterations control the number of iteration the algorithm run, the user choose it.
        global number_of_iterations
        number_of_iterations=0
        if self.iteration_string == 'Low':
            number_of_iterations=100
        elif self.iteration_string == 'Medium':
            number_of_iterations=600
        else:
            number_of_iterations=1200

        #resulotion control the output image resulotion, the user choose it.
        resolution = 0
        if self.resolution_string == 'Small- 256 px':
            resolution = 256
        elif self.resolution_string == 'Medium- 512 px':
            resolution = 512
        elif self.resolution_string  == 'Large- 1024 px':
            resolution = 1024

        #model type
        global modelType
        if self.model_string == 'Vgg16':
            modelType = 16
        elif self.model_string == 'Vgg19':
            modelType = 19

        # output_image get the result from the StyleMakerFunc.
        output_image = self.StyleMakerFunc(content_path, style_path, number_of_iterations, resolution, modelType)
        pixmap = QtGui.QPixmap(output_image.toqpixmap())
        pixmap = pixmap.scaledToHeight(250)
        self.output_frame.setPixmap(pixmap)
        self.output_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.output_frame.show()
        self.save_button.show()

        global flag_finish_generate
        flag_finish_generate = 1
        self.home_btn.setEnabled(True)

    """StyleMakerFunc is the main function that running the image style transfer algorithm"""
    def StyleMakerFunc(self, content_path, style_path, number_of_iterations, resolution, modelType):
        import numpy as np
        from PIL import Image
        import tensorflow as tf
        import tensorflow.contrib.eager as tfe
        from tensorflow.python.keras.preprocessing import image as kp_image
        from tensorflow.python.keras import models

        # Eager execution is a flexible machine learning platform for research and experimentation.
        # Since we're using eager our model is callable just like any other function.
        tf.enable_eager_execution()
        print("Eager execution: {}".format(tf.executing_eagerly()))

        # define calculation running step to the external thread.
        self.update_prograssBar_value = external_run_prograssBar()
        self.update_prograssBar_value.countChanged.connect(self.progressBar.setValue)

        # Content layer for the feature maps
        content_layers = ['block5_conv2']

        # Style layer for the feature maps.
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        numer_of_content_layers = len(content_layers)
        numer_of_style_layers = len(style_layers)

        # load_img function get the path of the image,
        # resize it and broadcast the image array such that it has a batch dimension.
        def load_img(path_to_img):
            max_dim = resolution
            img = Image.open(path_to_img)
            long = max(img.size)
            scale = max_dim / long
            img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
            img = kp_image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            return img

        # load_and_process_img is charge on load the image into the vgg network.
        def load_and_process_img(path_to_img):
            img = load_img(path_to_img)
            if modelType == 16:
                img = tf.keras.applications.vgg16.preprocess_input(img)
            elif modelType == 19:
                img = tf.keras.applications.vgg19.preprocess_input(img)
            return img

        def deprocess_img(processed_img):
            x = processed_img.copy()
            if len(x.shape) == 4:
                x = np.squeeze(x, 0)
            assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                       "dimension [1, height, width, channel] or [height, width, channel]")
            if len(x.shape) != 3:
                raise ValueError("Invalid input to deprocessing image")

            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = x[:, :, ::-1]

            x = np.clip(x, 0, 255).astype('uint8')
            return x

        # get_model function load the VGG16 model and access the intermediate layers.
        # Returns: a Keras model that takes image inputs and outputs the style and content intermediate layers.
        def get_model():
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            # We load pretrained VGG Network, trained on imagenet data
            if modelType == 16:
                vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
            elif modelType == 19:
                vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False
            # Get output layers corresponding to style and content layers
            style_outputs = [vgg.get_layer(name).output for name in style_layers]
            content_outputs = [vgg.get_layer(name).output for name in content_layers]
            model_outputs = style_outputs + content_outputs
            # Build model
            return models.Model(vgg.input, model_outputs)

        # get_content_loss function calculate the content loss that is the
        # Mean Squared Error between the two feature representations matrices.
        def get_content_loss(base_content, target):
            return tf.reduce_mean(tf.square(base_content - target))

        # Calculate the gram matrix for the style representation.
        def gram_matrix(input_tensor):
            # Make the image channels
            channels = int(input_tensor.shape[-1])
            a = tf.reshape(input_tensor, [-1, channels])
            n = tf.shape(a)[0]
            gram = tf.matmul(a, a, transpose_a=True)
            return gram / tf.cast(n, tf.float32)

        # get the style loss by calculate the Mean Squared Error between the two gram matrices.
        # We rescale the loss at a layer that is given by the size of the feature map and the number of filters
        def get_style_loss(base_style, gram_target):
            height, width, channels = base_style.get_shape().as_list()
            gram_style = gram_matrix(base_style)
            return tf.reduce_mean(tf.square(gram_style - gram_target))

        """This function load and preprocess the content and style images from their path.
            Then it will feed the images through the network to achieve
            the outputs of the inner layers.
            Returns the style and the content features representation."""
        def get_feature_representations(model, content_path, style_path):
            # Load our images into the VGG Network
            content_image = load_and_process_img(content_path)
            style_image = load_and_process_img(style_path)

            # compute content and style features
            style_outputs = model(style_image)
            content_outputs = model(content_image)

            # Get the style and content feature representations from our model
            style_features = [style_layer[0] for style_layer in style_outputs[:numer_of_style_layers]]
            content_features = [content_layer[0] for content_layer in content_outputs[numer_of_style_layers:]]
            return style_features, content_features

        """This function compute the content loss, style loss and sum the total loss.
            the model we use will give us access to the interim layers."""
        def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
            style_weight, content_weight = loss_weights

            # Feed the original image through our model. This will give us the content and
            # style representations at the defined layers.
            model_outputs = model(init_image)

            style_output_features = model_outputs[:numer_of_style_layers]
            content_output_features = model_outputs[numer_of_style_layers:]

            style_score = 0
            content_score = 0

            # calculate and sum the style losses from all the defined layers
            # weight each contribution of each loss layer equally
            weight_per_style_layer = 1.0 / float(numer_of_style_layers)
            for target_style, comb_style in zip(gram_style_features, style_output_features):
                style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

            # sum the content losses from all layers
            # weight each contribution of each loss layer equally
            weight_per_content_layer = 1.0 / float(numer_of_content_layers)
            for target_content, comb_content in zip(content_features, content_output_features):
                content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

            style_score *= style_weight
            content_score *= content_weight

            # Get total loss = content loss+ style loss
            loss = style_score + content_score
            return loss, style_score, content_score

        # Compute gradients according to input image
        def compute_grads(cfg):
            with tf.GradientTape() as tape:
                all_loss = compute_loss(**cfg)
            total_loss = all_loss[0]
            return tape.gradient(total_loss, cfg['init_image']), all_loss

        """run _style_transfer is the main function in this algorithm, run the main loop of the backpropagation with gradient descent. ."""
        def run_style_transfer(content_path,
                               style_path,
                               number_of_iterations=1200,
                               content_weight=1e3,
                               style_weight=1e-2):
            # We don't train any layers of our model, so we set their trainable to false.
            model = get_model()
            for layer in model.layers:
                layer.trainable = False

            # Get the style and content feature representations (from our specified intermediate layers)
            style_features, content_features = get_feature_representations(model, content_path, style_path)
            gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

            # Set initial image
            init_image = load_and_process_img(content_path)
            init_image = tfe.Variable(init_image, dtype=tf.float32)
            # We use Adam Optimizer
            opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

            # Store our best result
            best_loss, best_img = float('inf'), None

            # Create config
            loss_weights = (style_weight, content_weight)
            cfg = {
                'model': model,
                'loss_weights': loss_weights,
                'init_image': init_image,
                'gram_style_features': gram_style_features,
                'content_features': content_features
            }

            norm_means = np.array([103.939, 116.779, 123.68])
            min_vals = -norm_means
            max_vals = 255 - norm_means

            # Main loop
            for i in range(number_of_iterations):
                global count
                count=i
                self.update_prograssBar_value.start()
                print("Iteration: {}".format(i))
                grads, all_loss = compute_grads(cfg)
                loss, style_score, content_score = all_loss
                opt.apply_gradients([(grads, init_image)])
                clipped = tf.clip_by_value(init_image, min_vals, max_vals)
                init_image.assign(clipped)

                if loss < best_loss:
                    # Update best loss and best image from total loss.
                    best_loss = loss
                    best_img = deprocess_img(init_image.numpy())

            return best_img, best_loss

        best, best_loss = run_style_transfer(content_path, style_path, number_of_iterations=number_of_iterations)
        image = Image.fromarray(best)
        return image

"""external_run_prograssBar class control the thread running the ProgressBar."""
class external_run_prograssBar(QThread):
    countChanged = pyqtSignal(int)

    def run(self):
        global count
        global number_of_iterations
        progressVal =((count + 1) / number_of_iterations) * 100
        self.countChanged.emit(progressVal)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = main_window_gui()
    sys.exit(app.exec_())