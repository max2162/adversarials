import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter
from tkinter import ttk
import adversarial_attack
import threading
import matplotlib
import tensorflow as tf
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def to_image(img):
    """
    This function is used to scale the pixel values of images to the interval [0,1]
    :param img: image to be scaled
    :return: scaled image
    """
    img = (img - img.min()) / (img.max() - img.min())
    return img


def load_and_format_image(path, size):
    """
    This method is used to load the image stored at path, scale it to size and to pixel values in [0,1],
    and converting it to a tf.Tensor with format NHWC.
    :param path: The path the image to be loaded is stored.
    :param size: The 2-dim. size of the image. An 224 by 224 rgb image has size (224,224) for example.
    :return: formatted and rescaled image as NHWC tensor
    """
    image = tf.keras.preprocessing.image.load_img(path)
    image = image.resize(size)
    preprocessed_img = tf.keras.preprocessing.image.img_to_array(image)
    preprocessed_img = tf.reshape(preprocessed_img, shape=(1,
                                                           preprocessed_img.shape[0],
                                                           preprocessed_img.shape[1],
                                                           preprocessed_img.shape[2]))
    return preprocessed_img


class MainGUI:
    """"
    This class is used to construct a graphical user interface to perform an
    adversarial attack on the given model.
    """""

    def __init__(self, net, size=(224, 224), default_path='ressources/default.png'):
        """
        Initialization method for MainGUI objects
        :param net: model to predict classes
        :param size: Needs to be adjusted to model. Defaulted to (224, 224) to use the VGG16 model trained on ImageNet
        :param default_path: Path to image, that should be displayed before an image is selected by user.
        """
        self.window = tkinter.Tk()
        self.window.title('Image Predictor')
        self.window.configure(background='#fff')
        self.window.resizable(0, 0)
        self.window.geometry('684x624')
        self.path = default_path
        self.size = size
        self.net = net
        self.input_image = tf.ones(shape=(1, self.size[0], self.size[1], 3)) * 0.5
        self.adv_image = None
        self.diff = None
        self.canvas = None
        # Create button frame, that holds all necessary buttons and input elements
        self.button_frame = tkinter.Frame(self.window, bg='#fff')
        # Create button, that opens file chooser
        self.choose_img_btn = tkinter.Button(self.button_frame, text='Choose image',
                                             command=self.click_choose_image_btn)

        # Create button to perform adversarial creation, button should enable after choosing target class
        self.create_adversarial_btn = tkinter.Button(self.button_frame, text='Choose adversarial',
                                                     command=self.click_adversarial_btn)

        # organize button_frame via grid geometry and pack it
        self.choose_img_btn.grid(row=0, column=0, sticky='we', padx=(0, 200))
        self.create_adversarial_btn.grid(row=0, column=1, sticky='we', padx=(200, 0))

        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.pack(fill='x')

        # create image_frame to hold all images, their labels and prediction plots
        self.image_frame = tkinter.Frame(self.window, bg='#fff')

        # load default image
        default_img = Image.open(self.path)
        default_img = default_img.resize(self.size, Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(default_img)

        # first image, representing input with corresponding label
        self.first_img_label = tkinter.Label(self.image_frame, text='Initial Image', bg='#fff')
        self.first_img_label.grid(row=2, column=0)
        self.first_img = tkinter.Label(self.image_frame, image=img_tk, bg='#fff')
        self.first_img.grid(row=1, column=0)
        self.first_img.bind('<Button>', lambda e: self.show_prediction(self.input_image, self.first_img))

        # second image, representing adversarial example with corresponding label
        self.second_img_label = tkinter.Label(self.image_frame, text='Adversarial Example', bg='#fff')
        self.second_img_label.grid(row=2, column=1)
        self.second_img = tkinter.Label(self.image_frame, image=img_tk, bg='#fff')
        self.second_img.grid(row=1, column=1)
        self.second_img.bind('<Button>', lambda e: self.show_prediction(self.adv_image, self.second_img))

        # third image, representing scaled difference between input and adversarial example with corresponding label
        self.third_img_label = tkinter.Label(self.image_frame, text='Normalized Diff.', bg='#fff').grid(row=2, column=2)
        self.third_img = tkinter.Label(self.image_frame, image=img_tk, bg='#fff')
        self.third_img.grid(row=1, column=2)

        label = tkinter.Label(self.image_frame, text='Click on an image to see VGG16\'s top 5 predictions', bg='#fff')
        label.grid(row=3, column=0, columnspan=3, sticky='nswe')

        self.image_frame.pack(fill='both')
        self.window.bind('<Control-s>', lambda e: self.save_adv_example())
        self.window.mainloop()

    def click_choose_image_btn(self):
        """
        This method gets called, when the choose image btn is pressed. A filedialog will be opened, to choose an
        image file (.png or .jpg) to be opened, displayed and predicted.
        """
        self.path = tkinter.filedialog.askopenfilename(initialdir='./files', title='Select Image',
                                                       filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg")))
        if self.path is None or len(self.path) == 0:
            return
        new_img = Image.open(self.path)
        new_img = new_img.resize((self.size[0], self.size[1]), Image.ANTIALIAS)
        new_img_tk = ImageTk.PhotoImage(new_img)
        self.first_img.configure(image=new_img_tk)
        # we have to store the image, because otherwise the garbage-collector would delete it
        self.first_img.image = new_img_tk
        self.input_image = load_and_format_image(self.path, self.size)
        self.show_prediction(self.input_image, self.first_img)

    def display_barh(self, labels, probabilities):
        """
        This method is used to display a horizontal bar graph representing labels and probabilities
        """
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=75)

        y_pos = np.arange(len(labels))

        ax.barh(y_pos, probabilities, align='center', color='blue')
        ax.set_yticks(y_pos)

        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Probability\n')
        ax.set_title('Estimation of VGG16')

        if self.canvas is not None:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, sticky='nswe')

    def labels_and_probabilities(self, image):
        """
        Returns labels and self.net's corresponding estimated probabilities.
        :param image: image for which probabilities should be estimated.
        :return: list of labels and list of corresponding probabilities.
        """
        predictions = [(label, probability) for _, label, probability in
                       decode_predictions(self.net.predict(image), top=5)[0]]
        labels = [a for a, _ in predictions]
        probabilities = [b for _, b in predictions]
        return labels, probabilities

    def show_prediction(self, image, selected_label):
        """
        Displays prediction for top 5 classes in a barh-plot.
        :param image: image for which probabilities should be estimated.
        :param selected_label: Label which contains the image. This label will be visually highlighted in GUI
        :return: None
        """
        self.first_img.config(bg='#fff')
        self.second_img.config(bg='#fff')
        selected_label.config(bg='#06f')
        if image is None:
            return
        labels, probabilities = self.labels_and_probabilities(image)
        self.display_barh(labels, probabilities)

    def click_adversarial_btn(self):
        """
        This method gets called, when the adversarial image btn is pressed. A filedialog will be opened, to choose an
        image file (.png or .jpg) to be opened, displayed and predicted.
        """
        self.path = tkinter.filedialog.askopenfilename(initialdir='./files', title='Select Image',
                                                       filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg")))
        if self.path is None or len(self.path) == 0:
            return
        new_img = Image.open(self.path)
        new_img = new_img.resize((self.size[0], self.size[1]), Image.ANTIALIAS)
        new_img_tk = ImageTk.PhotoImage(new_img)
        self.second_img.configure(image=new_img_tk)
        self.second_img.image = new_img_tk
        self.adv_image = load_and_format_image(self.path, self.size)
        self.show_prediction(self.adv_image, self.second_img)

        dif = (self.input_image - self.adv_image).numpy()
        dif = np.reshape(dif, (self.size[0], self.size[1], 3))
        dif = to_image(dif)
        dif = dif * 255
        dif = np.uint8(dif)
        new_dif_img_tk = ImageTk.PhotoImage(Image.fromarray(dif))
        self.third_img.configure(image=new_dif_img_tk)
        self.third_img.image = new_dif_img_tk


def main():
    model = tf.keras.applications.VGG16(weights='imagenet')
    MainGUI(model)


if __name__ == '__main__':
    main()
