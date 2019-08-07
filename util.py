import os, pickle


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


def load_model(path, model_type, model_name='model', weights_file='weights'):
    '''
    Loads trained model.

    :param model_type: the type of the model, for example - char2vec_classification (should be an object)
    :param weights_file: the name of the weights file
    :param model_name: the name of the model file
    :param path: str, loads model from `path`.

    :return c2v_model: Chars2Vec object, trained model.
    '''

    path_to_model = path

    with open(f'{path_to_model}/{model_name}.pkl', 'rb') as f:
        structure = pickle.load(f)
        emb_dim, char_to_ix = structure[0], structure[1]

    c2v_model = model_type(emb_dim, char_to_ix)
    c2v_model.classifier_model.load_weights(f'{path_to_model}/{weights_file}.h5')
    c2v_model.classifier_model.compile(optimizer='adam', loss='mae')

    return c2v_model
