import matplotlib
import datetime
import time as t
from adversarial_attack_algorithms import fast_gradient_sign, carlini_wagner, one_pixel_attack


class AdversarialAttack:
    """
    This class serves as a wrapper class for easier access to the adversarial attacks implemented in
    adversarial_attack_algorithms package.
    """

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        # set selected algorithm to one of  {fast_gradient_sign_method, carlini_wagner, one_pixel_attack}
        attack_algorithm_name = kwargs['attack_algorithm']

        if attack_algorithm_name in ['fgsm', 'fast_gradient_sign_method']:
            self.attack_algorithm = fast_gradient_sign
        elif attack_algorithm_name in ['cw', 'carlini_wagner']:
            self.attack_algorithm = carlini_wagner
            # the carlini wagner attack relies on using logits, so the softmax layer has to be split
            self.model = self.attack_algorithm.preprocess_model(self.model)
        elif attack_algorithm_name in ['opa', 'one_pixel_attack']:
            self.attack_algorithm = one_pixel_attack
            # the carlini wagner attack relies on using logits, so the softmax layer has to be split

    def load_and_format_img(self, path, size):
        """
        As the attack algorithms of the adversarial_attack_algorithms package differ in how an image needs to be
        prepared this method calls the load_and_format_img method of the corresponding attack algorithm.
        :param path: The path the image to be loaded is stored.
        :param size: The 2-dim. size of the image. An 224 by 224 rgb image has size (224,224) for example.
        :return: the properly formatted image as tf.Tensor or np.ndarray
        """
        return self.attack_algorithm.load_and_format_image(path, size)

    def perform_attack(self, input_image, *args, **kwargs):
        """
        This method performs the actual attack. When in doubt which keyword arguments to pass, refer to the
        implementation of the attack.
        :param input_image: Every attack_algorithm gets passed an input image that is to be perturbated
        :param kwargs: Depending on the selected algorithm the keyword arguments differ
        :return: returns a adversarially perturbated image
        """
        return self.attack_algorithm.perform_attack(self.model, input_image, *args, **kwargs)

    @staticmethod
    def save_img_with_timestamp(image, directory, name=""):
        """
        This method can be used to save images with a timestamp included in the filename.
        The image will be saved as directory/name_timestamp.png .
        :param image: The image to be saved
        :param directory: string representing the path to the directory it should be saved in.
        :param name: any additional name to be added before timestamp
        """
        timestamp = datetime.datetime.fromtimestamp(t.time()).strftime('%Y_%m_%d_%H_%M_%S')
        # add an underscore if a name was given, to make it easier to read
        name = name if name == '' else name + '_'
        matplotlib.image.imsave('{}/{}{}.png'.format(directory, name, timestamp), image)
