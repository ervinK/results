from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json

import numpy as np

batch_size = 30  
epochs = 250  
latent_dim = 300  # Latent dimensionality of the encoding space.
num_samples = 30000  

data_path = 'rhyme_3_legnth_2_3.txt' # tanitoadatbazis

input_texts = [] #bemeneti szovegek
target_texts = [] #kimeneti szovegek
input_characters = set()  #lehetseges bemeneti karakterek
target_characters = set()  #lehetseges kimeneti karakterek


with open(data_path, 'r', encoding='utf-8') as f: #beolvasas: be \t ki \n formatumban soronkent
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    # spliteljük a lines tombot ugy, hogy ha a num_sample nagyobb, mint a tenyleges hossz, akkor se tortenjen problema
    input_text, target_text = line.split('\t')
    
    #szettorjuk a sort \t menten

    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

    #hozzaadjuk 

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters)) #bemeneti karakter map
target_characters = sorted(list(target_characters)) #kimeneti karakter map


num_encoder_tokens = len(input_characters) # bemeneti karakterek szama
num_decoder_tokens = len(target_characters) # kimeneti karakterek szama

max_encoder_seq_length = max([len(txt) for txt in input_texts]) # leghosszabb bemeneti szoveg hossza
max_decoder_seq_length = max([len(txt) for txt in target_texts]) # leghosszabb kimeneti szoveg hossza

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

# input_token_index: a bemeneti tokenek indexei, 0: 'a', 1: 'b' ... formatumban

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

#a kimeneti tokenek indexei, 0: 'a', 1: 'b' ... formatumban


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


"""
m*n-es matrixot hozunk letre, ahol:
- m sorok a leghosszabb karakterlanc a db-ben
- n oszlopok a szótár karaktereinek száma
"""

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    # vegigiteralunk az encoder matrixokon es a megfelelo sorokban 1-est rakunk a megfelelo karakterhez
    for t, char in enumerate(target_text):
        # vegigiteralunk a decoder matrixokon es a megfelelo sorokban 1-est rakunk a megfelelo karakterhez
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            #a decodernek van target dataja is, ami elorebb jar egyel, de ugyanazt tartalmazza
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

 
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


