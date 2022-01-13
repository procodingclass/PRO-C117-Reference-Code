import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


train_data = pd.read_excel("./static/assets/data_files/text-emotion-training-dataset.xlsx")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text_Emotion"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model_files/Text_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

emo_code_url = {
    "anger": [0, "./static/assets/emoticons/Anger.png"],
    "fear": [1,"./static/assets/emoticons/Fear.png" ],
    "joy": [2, "./static/assets/emoticons/Joy.png"],
    "love": [3, "./static/assets/emoticons/Love.png"],
    "sadness": [4, "./static/assets/emoticons/Sadness.png"],
    "surprise": [5, "./static/assets/emoticons/Surprise.png"]
    }

def predict(text):

    predicted_emotion=""
    predicted_emotion_img_url=""
    
    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=1)        
        print(predicted_class_label)   
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url