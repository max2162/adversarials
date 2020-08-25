import tensorflow as tf
import numpy as np


def load_and_format_image(path, size):
    """
    This method is used to load the image stored at path, scale it to size and to pixel values in {0,1,...,255},
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


def perform_attack(model, input_image, epsilon):
    """
    Method to perform Fast Gradient  Sign Method to perturbate an image to generate an adversarial example.
    :param model: model to generate an adversarial example for
    :param input_image: image to be perturbated
    :param epsilon: parameter controlling for amount of change
    :return: perturbated image
    """
    labels = model.predict(input_image)[0]
    actual_label = np.argmax(labels)
    actual_label_one_hot = np.zeros(len(labels))
    actual_label_one_hot[actual_label] = 1

    with tf.GradientTape() as t:
        t.watch(input_image)
        prediction = tf.dtypes.cast(model(input_image)[0], tf.float64)
        loss_value = tf.keras.losses.categorical_crossentropy(actual_label_one_hot, prediction)
    grad = t.gradient(loss_value, input_image)
    gradient_sign = tf.sign(grad)
    perturbated_image = input_image + epsilon * gradient_sign
    image_shape = (input_image.shape[1], input_image.shape[2], input_image.shape[3])

    perturbated_image_rescaled = np.uint8(np.reshape(perturbated_image.numpy(), image_shape))
    return perturbated_image_rescaled
