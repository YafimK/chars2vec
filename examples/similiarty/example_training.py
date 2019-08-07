import chars2vec_classification
import string
import util

dim = 50

path_to_model = 'path/to/model/directory'

X_train = [('mecbanizing',),
           ('dicovery'),
           ('prot$oplasmatic'),
           ('copulateng',),
           'estry',
           'cirrfosis'
           ]

y_train = [0, 0, 0, 1, 1, 1]

model_chars = string.printable

# Create and train chars2vec_classification model using given training data
my_c2v_model = chars2vec_classification.train_model(dim, X_train, y_train, model_chars, launch_tensorboard=True,
                                                    verobsity=1)

# Save pretrained model
util.save_model(my_c2v_model, path_to_model)

words = ['list', 'of', 'words']

# Load pretrained model, create word embeddings
c2v_model = util.load_model(path_to_model, chars2vec_classification)
word_embeddings = c2v_model.vectorize_words(words)
