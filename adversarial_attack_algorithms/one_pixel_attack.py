import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib
import datetime
import time as t
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input


def generate_agents(population_size=400, size=(224, 224)):
    """
    This method is used to generate an initial population of agents.
    An agent is a 5-tuple (x, y, r, g, b) where x and y are the x- and y-coordinates
    of the pixel and r, g and b are the respective new values for the three color channels of
    an rgb image.
    :param size: size of the image. Default is set to (224, 224) as this is the case for VGG16
    :param population_size: number of agents in population. Needs to be at least 4.
    :return: returns a list of size population_size containing 5-tuples
    """
    xs = np.random.randint(0, size[0], size=population_size)
    ys = np.random.randint(0, size[1], size=population_size)
    rs, gs, bs = np.random.normal(128, 127, size=[3, population_size])
    agents = [xs, ys, rs, gs, bs]
    agents = np.transpose(agents)
    return agents


def evaluate_fitness(model, input_image, agent, target):
    """
    This method is used to evaluate the fitness of a given agent. To determine the fitness, the model is used to predict
    the class of the input_image after it got perturbated by the agent. The agents fitness is equal to the probability
    the model predicted for the target class. When performing untargeted attack, the lower the fitness, the stronger the
    agent.
    :param model: model to be fooled
    :param input_image: input_image unperturbated
    :param agent: 5-tuple (x, y, r, g, b) with x,y coordinates and rgb pixel values of r, g and b
    :param target: class the input_image should be classified as. Low fitness => more likely to be misclassified
    :return: fitness value reflecting the probability of the image to be predicted as of class target
    """
    perturbated_image = generate_perturbated_image(input_image, agent)
    return model.predict(perturbated_image)[0][target]


def generate_perturbated_image(input_image, agent=None):
    """
    This method applies the changes encoded in an agent 5-tuple (x, y, r, g, b) to an image and then
    reshapes it into an NHWC tensor to be able to feed it into a keras model
    :param input_image: initial unperturbated image
    :param agent: 5-tuple (x, y, r, g, b) with x,y coordinates and rgb pixel values of r, g and b
    :return: NHWC tensor representing perturbated image
    """
    perturbated_image = input_image.copy()
    if agent is not None:
        x, y, r, g, b = agent
        x = max(min(x, 223), 0)
        y = max(min(y, 223), 0)
        perturbated_image[int(x)][int(y)] = [r, g, b]
    image_reshaped = tf.reshape(perturbated_image, shape=(1, 224, 224, 3))
    return image_reshaped


def mutate_population(population, fitness):
    """
    This method applies differential evolution to the population and returns a mutated population with strictly better
    fitness values (in case of untargeted attack lower) than the input population.
    :param population: list of agent 5-tuples (x, y, r, g, b)
    :param fitness: list of fitness of corresponding agents in population
    :return: (population, fitness) with corresponding updated population and fitness
    """
    for agent_index in range(len(population)):
        r1, r2, r3 = random.sample(range(population_size), 3)
        y = [0, 0, 0, 0, 0]
        for i in range(5):
            y[i] = population[r1][i] + factor * (population[r2][i] - population[r3][i])
        fitness_y = evaluate_fitness(model, input_image, y, target)
        if fitness_y <= fitness[agent_index]:
            fitness[agent_index] = fitness_y
            population[agent_index] = y
    return population, fitness


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
    return preprocessed_img


def perform_attack(model, input_image, population_size, iterations):
    """
    This method is used to perform the One Pixel Attack. Where input_image is tried to be perturbated by only changing
    a single pixel in order to be misclassified by model.
    :param model: model to be fooled
    :param input_image: initial unperturbated image (np.ndarray with pixel channel values in {0,1,..,255})
    :param population_size: Large values affect performance, but yield better results.
    :param iterations: Number of iterations to perform. Population is mutated according to differential evolution
                       every iteration
    :return: best perturbated_image. Not guaranteed to fool model.
    """
    reformatted_input_img = generate_perturbated_image(input_image)
    target = np.argmax(model.predict(reformatted_input_img)[0])

    population = generate_agents(population_size)
    # fitness is a list of the same size as population, containing the corresponding fitness values
    fitness = [evaluate_fitness(model, input_image, agent, target) for agent in population]

    for iteration in range(iterations):
        population, fitness = mutate_population(population, fitness)
        if min(fitness) < 0.05:
            break

    min_index = np.argmin(fitness)
    best_agent = population[min_index]
    perturbated_image = generate_perturbated_image(input_image, best_agent)

    return perturbated_image
