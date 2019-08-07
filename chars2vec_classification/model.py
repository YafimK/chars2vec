import numpy as np
import pickle
import keras
import os


class Chars2Vec_classifier:

    def __init__(self, emb_dim, char_to_ix):
        '''
        Creates chars2vec_classification model.

        :param emb_dim: int, dimension of embeddings.
        :param char_to_ix: dict, keys are characters, values are sequence numbers of characters.
        '''

        if not isinstance(emb_dim, int) or emb_dim < 1:
            raise TypeError("parameter 'emb_dim' must be a positive integer")

        if not isinstance(char_to_ix, dict):
            raise TypeError("parameter 'char_to_ix' must be a dictionary")

        self.char_to_ix = char_to_ix
        self.ix_to_char = {char_to_ix[ch]: ch for ch in char_to_ix}
        self.vocab_size = len(self.char_to_ix)
        self.dim = emb_dim
        self.cache = {}

        # Embedding model
        self.embedding_model = keras.Sequential()
        self.embedding_model.add(keras.layers.Input(shape=(None, self.vocab_size)))
        self.embedding_model.add(keras.layers.LSTMCell(emb_dim, return_sequences=True))
        self.embedding_model.add(keras.layers.LSTMCell(emb_dim))

        # Classification model
        self.classifier_model = keras.Sequential()
        self.classifier_model.add(keras.Input(shape=(None, self.vocab_size)))
        self.classifier_model.add(self.embedding_model)
        self.classifier_model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
        self.classifier_model.compile(optimizer='adam', loss='mae')

    def fit(self, words, targets,
            max_epochs, patience, validation_split, batch_size):
        '''
        Fits model.

        :param words: list or numpy.ndarray of words.
        :param targets: list or numpy.ndarray of targets.
        :param max_epochs: parameter 'epochs' of keras model.
        :param patience: parameter 'patience' of callback in keras model.
        :param validation_split: parameter 'validation_split' of keras model.
        :param batch_size: parameter 'batch_size' of keras model.
        '''

        if not isinstance(words, list) and not isinstance(words, np.ndarray):
            raise TypeError("parameters 'word_pairs' must be a list or numpy.ndarray")

        if not isinstance(targets, list) and not isinstance(targets, np.ndarray):
            raise TypeError("parameters 'targets' must be a list or numpy.ndarray")

        x_1 = []

        for word in words:
            emb_list_1 = []

            if not isinstance(word, str):
                raise TypeError("word must be a string")

            word = word.lower()

            for t in range(len(word)):

                if word[t] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[word[t]]] = 1
                    emb_list_1.append(x)

                else:
                    emb_list_1.append(np.zeros(self.vocab_size))

            x_1.append(np.array(emb_list_1))

        x_1_pad_seq = keras.preprocessing.sequence.pad_sequences(x_1)

        self.classifier_model.fit([x_1_pad_seq], targets,
                                  batch_size=batch_size, epochs=max_epochs,
                                  validation_split=validation_split,
                                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)])

    def vectorize_words(self, words, maxlen_padseq=None):
        '''
        Returns embeddings for list of words. Uses cache of word embeddings to vectorization speed up.

        :param words: list or numpy.ndarray of strings.
        :param maxlen_padseq: parameter 'maxlen' for keras pad_sequences transform.

        :return word_vectors: numpy.ndarray, word embeddings.
        '''

        if not isinstance(words, list) and not isinstance(words, np.ndarray):
            raise TypeError("parameter 'words' must be a list or numpy.ndarray")

        words = [w.lower() for w in words]
        unique_words = np.unique(words)
        new_words = [w for w in unique_words if w not in self.cache]

        if len(new_words) > 0:

            list_of_embeddings = []

            for current_word in new_words:

                if not isinstance(current_word, str):
                    raise TypeError("word must be a string")

                current_embedding = []

                for t in range(len(current_word)):

                    if current_word[t] in self.char_to_ix:
                        x = np.zeros(self.vocab_size)
                        x[self.char_to_ix[current_word[t]]] = 1
                        current_embedding.append(x)

                    else:
                        current_embedding.append(np.zeros(self.vocab_size))

                list_of_embeddings.append(np.array(current_embedding))

            embeddings_pad_seq = keras.preprocessing.sequence.pad_sequences(list_of_embeddings, maxlen=maxlen_padseq)
            new_words_vectors = self.classifier_model.predict([embeddings_pad_seq])

            for i in range(len(new_words)):
                self.cache[new_words[i]] = new_words_vectors[i]

        word_vectors = [self.cache[current_word] for current_word in words]

        return np.array(word_vectors)


def save_model(c2v_model, path_to_model, model_name="model"):
    '''
    Saves trained model to directory.

    :param model_name: the name to use for the model file.
    :param c2v_model: Chars2Vec object, trained model.
    :param path_to_model: str, path to save model.
    '''

    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    c2v_model.embedding_model.save_weights(path_to_model + '/weights.h5')

    with open(f'{path_to_model}/{model_name}.pkl', 'wb') as f:
        pickle.dump([c2v_model.dim, c2v_model.char_to_ix], f, protocol=2)


def load_model(path, model_name='model', weights_file='weights'):
    '''
    Loads trained model.

    :param path: str, loads model from `path`.

    :return c2v_model: Chars2Vec object, trained model.
    '''

    path_to_model = path

    with open(f'{path_to_model}/{model_name}.pkl', 'rb') as f:
        structure = pickle.load(f)
        emb_dim, char_to_ix = structure[0], structure[1]

    c2v_model = Chars2Vec_classifier(emb_dim, char_to_ix)
    c2v_model.classifier_model.load_weights(f'{path_to_model}/{weights_file}.h5')
    c2v_model.classifier_model.compile(optimizer='adam', loss='mae')

    return c2v_model


def train_model(emb_dim, X_train, y_train, model_chars,
                max_epochs=200, patience=10, validation_split=0.05, batch_size=64):
    '''
    Creates and trains chars2vec_classification model using given training data.

    :param emb_dim: int, dimension of embeddings.
    :param X_train: list or numpy.ndarray of words.
    :param y_train: list or numpy.ndarray of target values that describe if this word matches the binary question
    :param model_chars: list or numpy.ndarray of basic chars in model.
    :param max_epochs: parameter 'epochs' of keras model.
    :param patience: parameter 'patience' of callback in keras model.
    :param validation_split: parameter 'validation_split' of keras model.
    :param batch_size: parameter 'batch_size' of keras model.

    :return c2v_model: Chars2Vec object, trained model.
    '''

    if not isinstance(X_train, list) and not isinstance(X_train, np.ndarray):
        raise TypeError("parameter 'X_train' must be a list or numpy.ndarray")

    if not isinstance(y_train, list) and not isinstance(y_train, np.ndarray):
        raise TypeError("parameter 'y_train' must be a list or numpy.ndarray")

    if not isinstance(model_chars, list) and not isinstance(model_chars, np.ndarray):
        raise TypeError("parameter 'model_chars' must be a list or numpy.ndarray")

    char_to_ix = {ch: i for i, ch in enumerate(model_chars)}
    c2v_model = Chars2Vec_classifier(emb_dim, char_to_ix)

    targets = [float(el) for el in y_train]
    c2v_model.fit(X_train, targets, max_epochs, patience, validation_split, batch_size)

    return c2v_model
