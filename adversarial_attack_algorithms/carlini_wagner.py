import tensorflow as tf
import numpy as np


def loss(model, perturbated_image, target):
    """
    This method is used to calculate the loss function specified by carlini and wagner to be the most promising
    to use for this attack.
    See: https://ieeexplore.ieee.org/abstract/document/7958570 for more information
    :param model: attacked model
    :param perturbated_image: adversarial example
    :param target: target class the perturbated_image should be misclassified as.
    :return: loss value
    """
    # model(perturbated_image) returns [output, output_pre_activation] with output and output_pre_activation
    # of shape [1 1000], so indexing by [1][0] returns output_pre_activation in shape [1000]
    output_pre_activation = model(perturbated_image)[1][0]
    max_z = tf.reduce_max(output_pre_activation)
    z_t = output_pre_activation[target]
    return tf.maximum(0.0, max_z - z_t)


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
    # the carlini-wagner attack generates images with values in [0,1], so the loaded image with values in
    # {0,1,..,255} needs to be rescaled
    preprocessed_img = preprocessed_img / 255.0
    preprocessed_img = tf.reshape(preprocessed_img, shape=(1,
                                                           preprocessed_img.shape[0],
                                                           preprocessed_img.shape[1],
                                                           preprocessed_img.shape[2]))
    return preprocessed_img


def preprocess_model(model):
    """
    The carlini-wagner attack relies on having an option to obtain the logits that serve as input to the
    softmax layer, so the last layer of the model has to be split.
    :param model: model which softmax layer has to be split
    :return: model with split softmax layer
    """
    model.layers[-1].activation = tf.identity
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    # As the carlini wagner attack considers images to have pixel values in [0,1], but most image processing models
    # use 8-bit color channels, we need to rescale by a factor of 255.
    # We do not apply any rounding here, as this would create artificial plateaus that would worsen the results
    # obtained by Adam optimizer.
    inputs_scaled = inputs * 255
    output_pre_softmax = model(inputs_scaled)
    output = tf.nn.softmax(output_pre_softmax)
    return tf.keras.Model(inputs=inputs, outputs=[output, output_pre_softmax])


@tf.function
def perform_iteration(opt, model, input_img, target, w, c):
    """
    This method implements a single update step using the optimizer opt, given the model, input image, target,
    the learnable perturbation matrix w and the tradeoff variable c.
    :param opt: in their paper carlini and wagner use the Adam optimizer
    :param model: model to be attacked
    :param input_img: image to be perturbated
    :param target: target the perturbated image should be misclassified as
    :param w: learnable perturbation value
    :param c: tradeoff between distance and loss. High c means higher success rate, lower c means increased quality of
              adversarial example
    :return: total_loss and loss_pre_softmax
    """
    with tf.GradientTape() as tape:
        tape.watch(w)
        delta = (1 / 2) * (tf.tanh(w) + 1) - input_img
        loss_pre_softmax = loss(model, input_img + delta, target)
        loss_distance = tf.norm(delta, ord='euclidean')
        total_loss = loss_distance + c * tf.dtypes.cast(loss_pre_softmax, tf.dtypes.float32)
    gradients = tape.gradient(total_loss, w)
    opt.apply_gradients(zip([gradients], [w]))
    return total_loss, loss_pre_softmax


def to_img(input_image, w):
    """
    This method is used to generate an image with pixel values in {0,1,..,255} represented as 4d np.ndarray.
    :param input_image: unperturbated image
    :param w: perturbation encoded in w
    :return: image with pixel values in {0,1,..,255} represented as np.ndarray
    """
    delta = (1 / 2) * (tf.tanh(w) + 1) - input_image
    output_img = input_image + delta
    image_shape = (-1, input_image.shape[1], input_image.shape[2], input_image.shape[3])
    output_img_rescaled = output_img * 255
    return np.uint8(np.reshape(output_img_rescaled.numpy(), image_shape))


def summary_entry(file_writer, epoch, input_image, w, misclassification_loss, loss_val):
    with file_writer.as_default():
        if epoch % 100 == 0:
            # to save disk space only every 100 steps a image is written to the summary
            tf.summary.image("Perturbated Image", to_img(input_image, w), step=epoch)
        tf.summary.scalar("Misclassification Loss", misclassification_loss, step=epoch)
        tf.summary.scalar("Total Loss", loss_val, step=epoch)


def perform_attack(model, input_image, target, epochs=2500, c=1, tb=True, tb_path='./logs/cw'):
    """
    This method is used to perform the actual carlini-wagner attack. Where input_image is tried to be perturbated in such
    way, that model misclassifies the perturbated image as instance of target.
    :param model: model to be fooled
    :param input_image: image to be perturbated (NHWC Tensor)
    :param target: int representing class
    :param epochs: amount of perturbation steps (2500 seems empirically to be enough when c=1)
    :param c: tradeoff parameter between misclassification loss and distance loss
    :param tb: if True a TensorBoard log will be created
    :param tb_path: path for TensorBoard log
    :return: np.ndarray with pixel values in {0,1,...,255}
    """
    # w is the parameter on which we perform learning steps, see to_img(input_image, w) to see how perturbation works.
    w = tf.Variable(tf.zeros(tf.shape(input_image)), dtype=tf.dtypes.float32)
    optimizer = tf.keras.optimizers.Adam()
    file_writer = tf.summary.create_file_writer(tb_path)
    for epoch in range(epochs):
        loss_val, misclassification_loss = perform_iteration(optimizer, model, input_image, target, w, c)
        if tb:
            summary_entry(file_writer, epoch, input_image, w, misclassification_loss, loss_val)

    return to_img(input_image, w)[0]
