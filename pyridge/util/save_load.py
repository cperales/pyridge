import pickle
import os


def save_classifier(clf, folder=None, name='classifier'):
    """
    Save the classifier object into a pickle.

    :param clf: classifier object.
    :param name: name of the pickle.
    :return:
    """
    name += '.pkl'
    if folder is not None:
        filename = os.path.join(folder, name)
    else:
        filename = name
    with open(filename, 'wb') as name_clf:
        pickle.dump(clf, name_clf)


def load_classifier(folder=None, name='classifier'):
    """
    Load the classifier object from a pickle.

    :param clf: classifier object.
    :param name: name of the pickle.
    :return:
    """
    name += '.pkl'
    if folder is not None:
        filename = os.path.join(folder, name)
    else:
        filename = name
    with open(filename, 'rb') as name_clf:
        clf = pickle.load(name_clf)
    return clf
