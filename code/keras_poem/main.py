import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint

DATA_FILE = 'rap.txt'
seq_length = 10
lstm_size = 10
dropout_rate = 0.2
epochs = 10
batch_size = 256
model_file = 'model/baseline_seq_{}_lstm_{}.h5'.format(seq_length, lstm_size)
start_id = 6666


def load_data():
    with open(DATA_FILE, 'r') as f:
        text = f.read()
        return text.lower()


def create_mapping(text):
    vocab = sorted(list(set(text)))
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    return vocab, idx_to_char, char_to_idx


def process_data(text, vocab, char_to_idx):
    x = []
    y = []
    length = len(text)
    # seq_length = 100
    for i in range(length - seq_length):
        sequence = text[i: i + seq_length]
        label = text[i + seq_length]
        x.append([char_to_idx[ch] for ch in sequence])
        y.append(char_to_idx[label])
    x_modified = np.reshape(x, (len(x), seq_length, 1))
    x_modified = x_modified / float(len(vocab))
    y_modified = utils.to_categorical(y, len(vocab))
    return x, y, x_modified, y_modified


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(lstm_size, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_size))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, x, y):
    # Load model if exists
    if Path(model_file).is_file():
        print('Model file {} found. Loading weights...'.format(model_file))
        model.load_weights(model_file)
    else:
        print('Model file {} not found. Training from scratch...'.format(model_file))
    # Define checkpoint
    checkpoint_filepath = 'checkpoint/ckpt-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, verbose=1)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    model.save_weights(model_file)


def sample(model, x, vocab, idx_to_char):
    # Load model
    model.load_weights(model_file)
    # Init input and final prediction
    string_mapped = x[start_id]
    full_string = [idx_to_char[value] for value in string_mapped]
    for i in range(400):
        x = np.reshape(string_mapped, (1, len(string_mapped), 1))
        x = x / float(len(vocab))
        # Feed in LSTM and sample
        pred_index = np.argmax(model.predict(x, verbose=0))
        full_string.append(idx_to_char[pred_index])
        # Update input
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]
    return ''.join(full_string)


def main():
    text = load_data()
    vocab, idx_to_char, char_to_idx = create_mapping(text)
    x, y, x_modified, y_modified = process_data(text, vocab, char_to_idx)
    model = create_model((x_modified.shape[1], x_modified.shape[2]), y_modified.shape[1])

    # Train
    train_model(model, x_modified, y_modified)

    # Generate
    txt = sample(model, x, vocab, idx_to_char)
    print('Generated text:', txt)


if __name__ == '__main__':
    main()
