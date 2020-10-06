import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from adversarial_attack import AdversarialAttack


def main():
    model = tf.keras.applications.VGG16()
    [(_, height, width, _)] = model.layers[0].input_shape
    size = height, width

    # path = 'files/images/ostrich.jpg'
    # aa = AdversarialAttack(attack_algorithm='carlini_wagner', model=model)
    # input_image = aa.load_and_format_img(path, size)
    # # target 100 meaning the target class is the black swan, see: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    # adv_img = aa.perform_attack(input_image, target=100, epochs=3000, tb_path='./logs/cw/ostrich2/')
    # aa.save_img_with_timestamp(adv_img, directory='files/adversarial_examples', name='cw_ostrich_3000_epochs_t_100')

    path = 'files/images/sea_lion.jpg'
    aa = AdversarialAttack(attack_algorithm='carlini_wagner', model=model)
    input_image = aa.load_and_format_img(path, size)
    # target 291 meaning the target class is the lion, see: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    adv_img = aa.perform_attack(input_image, target=291, epochs=5000, tb_path='./logs/cw/sea_lion/')
    aa.save_img_with_timestamp(adv_img, directory='files/adversarial_examples', name='cw_sea_lion_5000_epochs_t_291')

    # path = 'files/images/sea_lion.jpg'
    # aa = AdversarialAttack(attack_algorithm='fgsm', model=model)
    # input_image = aa.load_and_format_img(path, size)
    # adv_img = aa.perform_attack(input_image, epsilon=0.004)
    # aa.save_img_with_timestamp(adv_img, directory='files/adversarial_examples', name='sea_lion_fgsm_0.004_grad')

    # path = 'files/images/ostrich.jpg'
    # aa = AdversarialAttack(attack_algorithm='fgsm', model=model)
    # input_image = aa.load_and_format_img(path, size)
    # adv_img = aa.perform_attack(input_image, epsilon=0.004)
    # aa.save_img_with_timestamp(adv_img, directory='files/adversarial_examples', name='fgsm_ostrich_eps_0.004')


if __name__ == '__main__':
    main()
