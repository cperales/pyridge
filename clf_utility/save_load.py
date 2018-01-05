import pickle
import os


def save_classifier(clf, name='classifier'):
    """
    Save the classifier object into a pickle.

    :param clf: classifier object.
    :param name: name of the pickle.
    :return:
    """
    try:
        file_name = os.path.join(os.path.sep, 'saved_clf', name)
        name_clf = open(file_name, 'wb')
    except FileNotFoundError:
        name_clf = open(name, 'wb')
    pickle.dump(clf, name_clf)
    name_clf.close()


def load_classifier(clf, name='classifier'):
    """
    Load the classifier object from a pickle.

    :param clf: classifier object.
    :param name: name of the pickle.
    :return:
    """
    try:
        file_name = os.path.join(os.path.sep, 'saved_clf', name)
        name_clf = open(file_name, 'wb')
    except FileNotFoundError:
        name_clf = open(name, 'wb')
    pickle.load(name_clf)
    name_clf.close()
