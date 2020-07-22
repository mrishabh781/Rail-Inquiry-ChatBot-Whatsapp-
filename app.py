import nltk
from flask import Flask, request
from nltk.stem.lancaster import LancasterStemmer
from twilio.twiml.messaging_response import MessagingResponse
nltk.download('punkt')
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import re
import requests

# apikey = 'your api key'
#for api key visit
#https://indianrailapi.com/


def train_route(number):
    url = "https://indianrailapi.com/api/v2/TrainSchedule/apikey/{}/TrainNumber/{}/".format(apikey, number)
    print(url)
    response = requests.get(url)
    response = response.json()
    x = ''
    try:
        for i in response['Route']:
            x = x + '\n' + i['StationName']
    except:
        return "error"
    return x


def check_pnr(pnr):
    url = "https://indianrailapi.com/api/v2/PNRCheck/apikey/{}/PNRNumber/{}/Route/1/".format(apikey, pnr)
    print(url)
    response = requests.get(url)
    response = response.json()
    print(response)
    result1_text = "Train Name - " + response['TrainName'] + "\n" + response['From'] + " To " + response[
        'To'] + "\n Journey Date - " + response['JourneyDate'] + "\n Chart Prepared - " + response['ChatPrepared']
    result_text = "Passenger No.     Booking status    Current Status\n\n"
    try:
        for i in response['Passangers']:
            result_text = result_text + i['Passenger'] + "         " + i['BookingStatus'] + "        " + i[
                'CurrentStatus'] + "\n\n"
    except:
        return "error"
    x = result1_text
    x = x + result_text
    return x


app = Flask(__name__)

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
print(data)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
print(data)
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(inp):
    inp = inp.lower()
    print(inp)
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    x = random.choice(responses)
    try:
        if 'route' in x:
            lst = re.findall('[0-9]+', inp)
            result = train_route(lst[0])
        elif 'PNR' in x:
            lst = re.findall('[0-9]+', inp)
            result = check_pnr(lst[0])
        else:
            result = x
    except:
        result = x

    return result


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/", methods=['POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Fetch the message
    msg = request.form.get('Body')
    result = chat(msg)
    # Create reply
    print(result)
    resp = MessagingResponse()
    resp.message(result)

    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
