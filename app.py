# importing flask modules
from flask import Flask, request, jsonify

#  Importing base Libraries to perform operations.
import numpy as np
np.set_printoptions(suppress=True)
import os

#  Calling necessary libraries from tensorflow network.
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from keras import initializers
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Dropout
from keras.initializers import Constant


# Importing Libraries for NLP
import re
import string
import spacy
spacy.prefer_gpu()
# to make spacy work in pipeline.
nlp_vocab = spacy.load('/home/affine/Downloads/en_core_web_sm-2.2.0/en_core_web_sm/en_core_web_sm-2.2.0', 
                       disable=['tagger', 'parser', 'ner'])
nlp_vocab.add_pipe(nlp_vocab.create_pipe('sentencizer'))



# Importing spellchecker & NLP
from spellchecker import SpellChecker


# lemmatization of words from spacy.
def spacy_lemmatize(x):
  x = nlp_vocab(x)
  x = [s.lemma_ for s in x]
  x = " ".join(x)
  return x
# Spelling collection
spell = SpellChecker()
def correct_spellings(x, spell=spell):
    """correct the misspelled words of a given corpus"""
    x = x.split()
    misspelled = spell.unknown(x)
    result = map(lambda word : spell.correction(word) if word in  misspelled else word, x)
    return " ".join(result)

# corpus cleaning. we will keep Lemmatization as False as it's time consuming activity we will later use it in parallel computing.
def corpus_cleaning(x, correct_spelling=True, remove_emojis=True, remove_stop_words=False, lemmatize=True):
    """Apply function to a clean a corpus"""
    x = x.lower().strip()
    # romove urls
    url = re.compile(r'https?://\S+|www\.\S+')
    x = url.sub(r'',x)
    # remove html tags
    html = re.compile(r'<.*?>')
    x = html.sub(r'',x)
    # remove punctuation
    operator = str.maketrans('','',string.punctuation) #????
    x = x.translate(operator)
    if correct_spelling:
        x = correct_spellings(x)
    if lemmatize:
        x = spacy_lemmatize(x)
    if remove_emojis:
        x = x.encode('ascii', 'ignore').decode('utf8').strip()
    return x


# loading the vectorizer again
import pickle

from_disk = pickle.load(open("vectorizer.pkl", "rb"))
loaded_vector = TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
loaded_vector.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
loaded_vector.set_weights(from_disk['weights'])


# Creating vocabulary with index values
voc = loaded_vector.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# Importing Word embedding
path_to_glove_file = os.path.join( "glove.6B.50d.txt")
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs


num_tokens = len(voc) + 2
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
# print("Converted %d words (%d misses)" % (hits, misses))


# creating embedding layer
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True,
)


#  Import K to clear session for model.
from keras import backend as K

LSTM_UNITS = 64
BATCH_SIZE = 128
DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS
EPOCHS = 100

#  Import K to clear session for model.
from keras import backend as K

# Padding of sentence is done 40
maxlen = 40


# Model 3
def BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 80, 30, 12], weight_decay=0.10):
    K.clear_session()
    x_input = Input(shape=(maxlen,))

    emb = Embedding(num_tokens,
                    embedding_dim,
                    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                    trainable=False, name='Embedding')(x_input)

    x = SpatialDropout1D(rate=spatialdropout, seed=10000)(emb)

    rnn = Bidirectional(
        LSTM(rnn_units, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed=123000),
             recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=123000)))(x)

    x1 = Conv1D(filters=filters[0], activation='relu', kernel_size=1, padding='same',
                kernel_initializer=initializers.glorot_uniform(seed=110000))(rnn)
    x2 = Conv1D(filters=filters[1], activation='relu', kernel_size=1, padding='same',
                kernel_initializer=initializers.glorot_uniform(seed=120000))(rnn)
    x3 = Conv1D(filters=filters[2], activation='relu', kernel_size=1, padding='same',
                kernel_initializer=initializers.glorot_uniform(seed=130000))(rnn)
    x4 = Conv1D(filters=filters[3], activation='relu', kernel_size=1, padding='same',
                kernel_initializer=initializers.glorot_uniform(seed=140000))(rnn)

    x1 = GlobalMaxPooling1D()(x1)
    x2 = GlobalMaxPooling1D()(x2)
    x3 = GlobalMaxPooling1D()(x3)
    x4 = GlobalMaxPooling1D()(x4)

    c = concatenate([x1, x2, x3, x4])
    x = Dense(256, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=111000))(c)
    x = Dropout(0.2, seed=10000)(x)
    x = BatchNormalization()(x)
    x_output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=110000))(x)

    model = Model(inputs=x_input, outputs=x_output)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(  # clipvalue=0.5,
        learning_rate=0.0001)  # clip value to avoid the gradient exploding
                  , metrics=['acc'])
    return model

### Loading saved model.
# loading the model params on which it was trained
load_model = BiLSTM_CNN()
#  loading the training weights back to model.
load_model.load_weights('BiLSTM_CNN_v5/content/BiLSTM_CNN/my_model')

# The below command creates WSGI application Web Services Gateway interface.
app = Flask(__name__)

# Default Page
@app.route(rule='/')
def default():
    return "System Running"


#### Page showing the results for HTML.
@app.route(rule='/api', methods=['POST'])
def api():
    # to get the input from the api 
    data = request.get_json(force=True)
    # converting json to input file
    predict_request = [corpus_cleaning(data['text'])]
    predict_request = loaded_vector(np.asarray([[s] for s in predict_request])).numpy()
    predict_request = tf.keras.preprocessing.sequence.pad_sequences(predict_request,maxlen= 40, padding='post')
    result = load_model.predict(predict_request)[0][0]
    return {'output' : str(result)}

if __name__ == '__main__':
    app.run(debug=True)  # the advantage of applying Degbug = True is that weblink auto refreshs with changes made in app.