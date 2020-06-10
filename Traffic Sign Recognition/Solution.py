import os
import random
import time
from typing import List, Tuple, Dict
import PIL
import matplotlib.pyplot as plt
import numpy
import sklearn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.ensemble import RandomForestClassifier

"""
Transform images to same size
"""

def resize_pad_image(image, size):
    """

    :param image: PIC.image
    :param size: desired size, example (20,20)
    :return: PIC.Image with padding and desired size
    """

    old_size = image.size
    width, height = old_size
    if width > height:
        desired_size = width
    else:
        desired_size = height
    new_size = (desired_size, desired_size)
    new_im = Image.new("RGB", new_size)

    new_im.paste(image, ((new_size[0] - old_size[0]),
                         (new_size[1] - old_size[1])))
    new_im = new_im.resize(size, Image.ANTIALIAS)
    return new_im


"""
Frequencies
"""

def find_freq(data: Dict[str, List[str]]):
    """
    :param data: dict(), where key is class label, value - list of images' filenames corresponding to that class
    :return: dict(), where key is class label, value - number of images occurrences  corresponding to that class
    """
    freq = dict()
    for a, b in data.items():
        freq[int(a)] = len(b)
    return freq

def plot_freq(freq):

    plt.figure()
    plt.title(f"Frequencies")
    plt.xlabel(f"class")
    plt.ylabel(f"num of images in training set")
    y = list()
    x = list()
    for a, b in freq.items():
        x.append(int(a))
        y.append(b)

    plt.bar(x, y)
    plt.savefig('output/plot_freq.png')

"""
Augmentation
"""
def change_image(image):
    """
    Create new picture by modifying brightness,contrast, blur
    :param image: PIL.image
    :return: changed PIL.image
    """
    brightness_param = random.uniform(0.3, 1.5)
    contrast_param = random.uniform(0.6, 1.5)
    blur_param = random.uniform(0, 1.7)
    brightness = PIL.ImageEnhance.Brightness(image)
    image = brightness.enhance(brightness_param)
    contrast = PIL.ImageEnhance.Contrast(image)
    image = contrast.enhance(contrast_param)

    image = image.filter(ImageFilter.GaussianBlur(radius=blur_param))

    return image

def augment(training_set_pictures, freq):
    """

    :param training_set_pictures:  dict(), where key is a class' label, value is a list of PIL.images corresponding to that class
    :param freq: dict(), where key is class label, value - number of images occurrences  corresponding to that class

    :return:augmented training_set_pictures
    """
    max = 0
    for a, b in freq.items():
        if b > max:
            max = b
    random_picture = None
    changed_random_picture = None
    for a, b in freq.items():
        length = len(training_set_pictures[a])
        for i in range(0, max - b):
            random_picture_index = random.randint(0, length - 1)
            random_picture = training_set_pictures[a][random_picture_index]
            changed_random_picture = change_image(random_picture)
            training_set_pictures[a].append(changed_random_picture)
    random_picture.save("output/before_augmentation.ppm")
    changed_random_picture.save("output/after_augmentation.ppm")

    return training_set_pictures
"""
Image normalization
What are the features?
"""

def format_images_for_model(images, size):
    """

    :param images: dict(), where key is a class' label, value is a list of PIL.images corresponding to that class
    :param size: desired size for each image
    :return: vector of features, vector of labels
    """
    padded_images = dict()
    images[0][0].save("output/original.ppm")
    for a, b in images.items():
        padded_images[a] = list()
        for i in b:
            padded_images[a].append(resize_pad_image(i, size))
    padded_images[0][0].save("output/padded_resized.ppm")
    labels = []
    data = []
    for a, b in padded_images.items():
        for i in b:
            vector_image = normalize_and_ravel_image(i)
            data.append(vector_image)
            labels.append(a)
    del padded_images
    return numpy.array(data), numpy.array(labels)

def normalize_and_ravel_image(image):
    """

    :param image: PIL.image
    :return: normalized vector
    """
    I = numpy.array(image)
    del image
    I = numpy.ravel(I, order='C')
    f = lambda x: x / 255
    normalized = f(I)
    return normalized

"""Evaluate"""

def get_examples_of_incorrectly_classified(predicted_y, true_y, data_val, number, size):
    """
    :param predicted_y: list of  y predicted by the model
    :param true_y: list of true values of y
    :param data_val: data which was put into model for testing/validation
    :param number: desired number of examples
    :param size: size of initial images
    :return: list of incorrectly classified images of length number, list of the true classes of these images,
     list of the classes which were wrongly predicted by the model

    """
    i = 0
    incorrectly_classified_indexes = list()
    incorrectly_classified = list()
    true_labels = list()
    predicted_labels = list()
    for a, b in zip(predicted_y, true_y):
        if a != b:
            incorrectly_classified_indexes.append(i)

        i = i+1
    sample = random.sample(range(0, len(incorrectly_classified_indexes)-1), number)
    for j in sample:
        predicted_class = predicted_y[j]
        real_class = true_y[j]
        image = from_byte_array_to_pic(size, data_val[j])
        incorrectly_classified.append(image)
        true_labels.append(real_class)
        predicted_labels.append(predicted_class)


    return incorrectly_classified, true_labels, predicted_labels


def plot_accuracy_vs_size(set_to_predict_name, accuracy_aug, accuracy_not_aug, sizes):
    """

    :param set_to_predict_name: testing set name in case we want to test model with several test sets and serve outputs accordingly
    :param accuracy_aug:  list of accuracies corresponding to sizes with augmentation
    :param accuracy_not_aug: list of accuracies corresponding to sizes without augmentation
    :param sizes: list of different sizes
    :return:
    """
    plt.figure()
    plt.title(f"Accuracy vs. Image size")
    plt.xlabel(f"sizes (px)")
    plt.ylabel(f"accuracy (sec)")
    plt.plot(sizes, accuracy_aug, label="augumented data", color="red")
    plt.plot(sizes, accuracy_not_aug, label="not augumented data", color="green")
    plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/accuracy_size.png")


def plot_precision_recall(set_to_predict_name, size, true_y, predicted_y_aug, predicted_y_no_aug):
    """

    :param set_to_predict_name: testing set name in case we want to test model with several test sets and serve outputs accordingly
    :param size: size of images model was trained by
    :param true_y: list of true values of y
    :param predicted_y_aug: list of predicted values of y with augmentation
    :param predicted_y_no_aug: list of predicted values of y without augmentation
    :return:
    """
    precision_aug, recall_aug, fbeta, support = sklearn.metrics.precision_recall_fscore_support(true_y, predicted_y_aug,
                                                                                        average=None, labels=range(0, 43))
    precision_not_aug, recall_not_aug, fbeta, support = sklearn.metrics.precision_recall_fscore_support(true_y, predicted_y_no_aug,
                                                                                        average=None, labels=range(0, 43))
    print("precision__aug", precision_aug)
    print("precision_not_aug", precision_not_aug)
    plt.figure()
    plt.title(f"Precision vs. Class Label with size {size}")
    plt.xlabel(f"Class")
    plt.bar(range(0, 43), precision_aug, label= "augmented data")
    plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/precision_size_aug_{size}.png")

    print("recall_aug", recall_aug)
    print("recall_not_aug", recall_not_aug)
    plt.figure()
    plt.title(f"Precision vs. Class Label with size {size}")
    plt.xlabel(f"Class")
    plt.bar(range(0, 43), precision_not_aug, label="not augmented data")
    plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/precision_size_not_aug_{size}.png")


    plt.figure()
    plt.title(f"Recall vs. Class Label with size {size}")
    plt.bar(range(0, 43), recall_aug, label="augmented data")
    plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/recall_size_aug_{size}.png")

    plt.figure()
    plt.title(f"Recall vs. Class Label with size {size}")
    plt.bar(range(0, 43), recall_not_aug, label="not augmented data")
    plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/recall_size_not_aug_{size}.png")





def plot_time_vs_size(set_to_predict_name, time_aug, time_not_aug, sizes):
    """

    :param set_to_predict_name: testing set name in case we want to test model with several test sets and serve outputs accordingly
    :param time_aug: list of times corresponding to sizes with augmentation
    :param time_not_aug: list of times corresponding to sizes without augmentation
    :param sizes: list of different sizes for testing
    :return:
    """
    plt.figure()
    plt.title(f"Time vs. Image size")
    plt.xlabel(f"sizes (px)")
    plt.ylabel(f"time (sec)")
    plt.plot(sizes, time_aug, label="augumented data", color="red")
    plt.plot(sizes, time_not_aug, label="not augumented data", color="green")
    # plt.legend()
    plt.savefig(f"output/{set_to_predict_name}/time_size.png")



"""
Tools
"""

def from_byte_array_to_pic(size, byte_array):
    """

    :param size: size of image
    :param byte_array: array image was converted to
    :return: PIL.Image
    """
    img = (byte_array.reshape((*size, 3)) * 255).astype("uint8")
    return PIL.Image.fromarray(img)

def return_top_level_dirs(path_to_dir: str) -> List[str]:
    """

    :param path_to_dir: path to the directory
    :return: list of names of only top level directories located in the directory which location is described by path_to_dir
    """
    direct = next(os.walk(path_to_dir))[1]
    return direct


def return_files_in_dir(path_to_dir: str) -> List[str]:
    """
    :param path_to_dir: path to the directory
    :return: list of paths to files stored in the directory which location is described by path_to_dir
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_to_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files


def turn_files_to_images(files_set):
    """

    :param files_set: dict(), where key is class label, value - list of images' filenames corresponding to that class
    :return:dict()  where key is class label, value - list of images  corresponding to that class
    """
    images_set = dict()
    for a, b in files_set.items():
        images_set[int(a)] = list()
        for path in b:
            image = Image.open(path)
            image.load()
            images_set[int(a)].append(image)
    return images_set


def get_num_of_tracks_for_class(files_in_dict: List[str]) -> int:
    """
    :param files_in_dict:  : dict(), where key is class label, value - list of images' filenames corresponding to that class
    :return: number of tracks in class
    """
    files = files_in_dict
    max = 0
    for j in files:
        if j.split(".")[-1] == "ppm":
            splitted = (j.split("/")[-1]).split("_")[0]
            if int(splitted) > max:
                max = int(splitted)
    return max + 1


def get_classes(path_to_dir):
    """

    :param path_to_dir: path to directory with classes
    :return: list of classes (in str format)
    """
    return return_top_level_dirs(path_to_dir)


def get_class_ppm_files(class_name, path_to_dir):
    """

    :param class_name: name of a class (str)
    :param path_to_dir: path to directory with classes
    :return: list of filenames with ppm extension
    """
    path_to_dir = path_to_dir + "/" + class_name
    files = return_files_in_dir(path_to_dir)
    ppm_files = []
    for file in files:
        if file.split(".")[-1] == "ppm":
            ppm_files.append(file)
    return ppm_files

def prepare_testing_images(path_to_dir):
    """

    :param path_to_dir: path to directory with testing images
    :return: dict()  where key is class label, value - list of images corresponding to that class
    """
    desctiption = open("GTSRB/GT-final_test.csv", "r")
    desctiption = desctiption.readlines()[1:]
    images:Dict[str, List[str]] = dict()
    for file in desctiption:
        parts = file.split(";")
        filename = path_to_dir + "/" + parts[0]
        class_label = int((parts[7]).split('\n')[0])
        if class_label not in images:
            images[class_label] = list()
        images[class_label].append(Image.open(filename))
    return images


"""
Split data
"""

def split_tracks_within_one_class(track_list):
    """

    :param track_list: list of tracks (str)
    :return: list of tracks for testing, list of tracks for training
    """
    testing_tracks = list()
    training_tracks = list()
    random.shuffle(track_list)
    for elem in range(len(track_list)):
        if elem >= int(len(track_list) * 0.8):
            training_tracks.append(track_list[elem])
        else:
            testing_tracks.append(track_list[elem])
    return testing_tracks, training_tracks


def split_files_within_one_class(cls, path_to_dir):
    """

    :param cls: classname
    :param path_to_dir: path to directory with classes
    :return: lists of files (images names) for training set, testing set
    """
    training_set_files = []
    testing_set_files = []
    cls_images = get_class_ppm_files(cls, path_to_dir)
    num_of_vids = get_num_of_tracks_for_class(cls_images)
    vids_list: List[List[str]] = list()
    for _ in range(num_of_vids):
        vids_list.append(list())
    for image in cls_images:
        vid_id = int((image.split("/")[-1]).split("_")[0])
        vids_list[vid_id].append(image)
    training_tracks, testing_tracks = split_tracks_within_one_class(vids_list)
    for track in testing_tracks:
        for image in track:
            testing_set_files.append(image)
    for track in training_tracks:
        for image in track:
            training_set_files.append(image)
    return training_set_files, testing_set_files


def split_images(path_to_dir):
    """

    :param path_to_dir:  path to directory with classes
    :return: list of images (PIL.Image) for training, list of images (PIL.Image) for testing, frequency dict(), where key is class label, value - number of images occurrences  corresponding to that class
    """
    classes: List[str] = get_classes(path_to_dir)
    training_files: Dict[str, List[str]] = dict()
    validation_files: Dict[str, List[str]] = dict()
    for cls in classes:
        training_files[cls], validation_files[cls] = split_files_within_one_class(cls, path_to_dir)
    freq = find_freq(training_files)
    validation_images = turn_files_to_images(validation_files)
    training_images = turn_files_to_images(training_files)
    return training_images, validation_images, freq







def main():

    #Create directory for the output
    try:
        os.mkdir(f"output")
    except:
        pass

    #Pick 5 sizes
    sizes = [(5, 5), (10, 10), (15,15), (30, 30), (45, 45)]

    #Pick number of examples which are incorrectly classified
    num_examples_incor_classified = 3

    #Choose special things (some plots will be made only with those parameters)
    special_aug_flags = [True, False]
    special_sizes = [(30, 30)]

    #Split
    training_images, validation_images, freq = split_images("GTSRB/Final_Training/Images")
    plot_freq(freq)

    #Prepare testing images
    test_images = prepare_testing_images("GTSRB/Final_Test/Images")

    #Choose test sets
    sets_for_prediction = {'validation':validation_images, 'test':test_images}

    #Create folder for each test set
    for name in sets_for_prediction:
        try:
            os.mkdir(f"output/{name}")
        except:
            pass

    for set_name, image_set_pred in sets_for_prediction.items():
        times = []
        accuracies = []
        predicted_y = []
        for augument_flag in [True, False]:
            accuracies_impl = []
            times_impl = []
            if augument_flag == True:
                training_images = augment(training_images, freq)
            for size in sizes:
                print(set_name, augument_flag, size)
                begin = time.monotonic()
                #Train model

                data_val, true_y = format_images_for_model(image_set_pred, size)
                data_tr, labels_tr,  = format_images_for_model(training_images, size)
                clf = RandomForestClassifier(n_jobs=-1,  n_estimators=50)
                clf.fit(data_tr, labels_tr)

                #Predict
                predicted_y_impl = clf.predict(data_val)
                predicted_ok = 0

                #Evaluate
                for a, b in zip(predicted_y_impl, true_y):
                    if a==b:
                        predicted_ok = predicted_ok + 1
                accuracy = predicted_ok/len(predicted_y_impl)
                print(accuracy)
                total = time.monotonic() - begin
                if size in special_sizes and augument_flag in special_aug_flags:
                    predicted_y.append(predicted_y_impl)

                    incorrectly_classified, true_labels, predicted_labels = get_examples_of_incorrectly_classified(predicted_y_impl, true_y, data_val, num_examples_incor_classified, size)
                    for i in range(len(incorrectly_classified)):
                        incorrectly_classified[i].save(f"output/{set_name}/incorrectly_classified_{i}.ppm")
                        example_of_guessed_class = image_set_pred[predicted_labels[i] - 1][0]
                        example_of_guessed_class.save(f"output/{set_name}/example_of_guessed_class_{i}.ppm")
                        example_of_true_class = image_set_pred[true_y[i] - 1][0]
                        example_of_true_class.save(f"output/{set_name}/example_of_true_class_{i}.ppm")

                times_impl.append(total)
                accuracies_impl.append(accuracy)
            times.append(times_impl)
            accuracies.append(accuracies_impl)
        plot_precision_recall(set_name, special_sizes[0], true_y, predicted_y[0], predicted_y[1])
        plot_time_vs_size(set_name,times[0], times[1], [j[0] for j in sizes])
        print(accuracies[0], accuracies[1])
        plot_accuracy_vs_size(set_name, accuracies[0], accuracies[1], [j[0] for j in sizes])





if __name__ == "__main__":
    main()
