from scraper import url_data_collect
from neural_network_helper import normalize_one_sample, preproces_attributes
from neural_network import get_model_prediction, load_model

means = [12345113.89037037, 12215814.084444445, 12145031.994074075, 0.1274074074074074, 0.5362962962962963, 7.41037037037037, 3.8533333333333335, 0.6607407407407407, 0.7614814814814815, 0.3896296296296296, 0.6607407407407407]
max = [20301027, 20201118, 20201117, 1, 7, 13, 10, 1, 1, 1, 1]
min = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

def format_url_input(url):
    if url.find("http")>=0:
        start = url.find("/")
        url = url[start + 2:]
    if url[0:4] == "www.":
        url = url[4:]
    return url

def format_attributes(attributes):
    # strip off string to colon
    for i in range (len(attributes)):
        attributes[i] = attributes[i][attributes[i].find(":") + 2:]
    num_attributes = preproces_attributes(attributes)
    norm_attributes = normalize_one_sample(num_attributes, means, min, max)
    return norm_attributes

def get_attributes(fullurl):
    url = format_url_input(fullurl)
    attributes = url_data_collect(url, fullurl=fullurl)
    norm_attributes = format_attributes(attributes)
    return norm_attributes

def predict():
    # input is a user defined url, prints the model's prediction
    model = load_model()

    fullurl = input("Please enter a url you would like to check\n")
    attributes = get_attributes(fullurl)
    phish, confidence = get_model_prediction(model, attributes)
    if phish:
        print("\nModel predicts this IS a phishing site with {}% confidence".format(str(float(confidence.data)*100)))
    else:
        print("\nModel predicts this IS NOT a phishing site with {}% confidence".format(str((1 - float(confidence.data))*100)))

if __name__ == '__main__':
    predict()