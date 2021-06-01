import os
import torch
import numpy as np

def extract_attributes_from_file(filename):
    # filename must be complete path
    f = open(filename, "r")
    contents = f.readlines()
    attributes = []
    for x in contents:
        attributes.append(x[x.find(":")+2:-1])
    return attributes[2:]


def format_date(date_string):
    numbers = date_string.replace("-", "")
    date_int = int(numbers)
    return date_int


def map_server_location(serverloc):
    filepath = os.path.join(os.getcwd(), "ML", "sever_locations")
    f = open(filepath, "r+")
    content = f.readlines()
    f.seek(0)
    servers = []
    mapping = None
    i = 0
    for line in content:
        line = line.strip("\n")
        if line.find(serverloc) >=0:
            mapping = i+1
        if line not in servers:
            f.write(line + "\n")
            servers.append(line)
        i += 1
    f.truncate()
    f.close()
    if mapping == None:
        mapping = i + 1
    return mapping


def preproces_attributes(attributes):
    # takes the attribute data and converts it to integers
    attributes[0] = format_date(attributes[0])
    attributes[1] = format_date(attributes[1])
    attributes[2] = format_date(attributes[2])
    attributes[3] = 0 if attributes[3] == 'not blacklisted' else 1
    attributes[4] = 0 if attributes[4] == "Nothing Found" or attributes[4] == "Nothing found" \
                         or attributes[4] == "Error" or attributes[4] == "" or attributes[4] == '' \
                            else int(attributes[4][:-3])
    attributes[5] = map_server_location(attributes[5])
    attributes[6] = 0 if attributes[6] == 'No pagerank found' or attributes[6] == "" \
                         or attributes[6] == "Error" else int(attributes[6][:-5])
    attributes[7] = 0 if attributes[7] == "False" else 1
    attributes[8] = 0 if attributes[8] == "False" else 1
    attributes[9] = 0 if attributes[9] == "False" else 1
    attributes[10] = 0 if attributes[10] == "False" else 1
    return attributes


def get_all_preprocessed(folder):
    # folder must be a complete path
    files = os.listdir(folder)
    all_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        num_attrs = preproces_attributes(string_attrs)
        all_data.append(num_attrs)
    return all_data


def normalize_samples(data):
    # subtract the means from min and max so the data will be 0 centered
    # then normalize so the data is between 0 and 1
    # used formula from
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

    means, min, max = get_means_min_max(data)
    print(means)
    print(max)
    print(min)
    for i in range(len(means)):
        min[i] -= means[i]
        max[i] -= means[i]

    for url_vector in data:
        for i in range(len(url_vector)):
            if((max[i] - min[i]) != 0):
                url_vector[i] = (url_vector[i] - means[i] - min[i])/(max[i] - min[i])
            else:
                url_vector[i] = 0
    return data


def normalize_one_sample(sample, means, min, max):
    # subtract the means from min and max so the data will be 0 centered
    # then normalize so the data is between 0 and 1
    for i in range(len(means)):
        min[i] -= means[i]
        max[i] -= means[i]
    for i in range(len(sample)):
        if ((max[i] - min[i]) != 0):
            sample[i] = (sample[i] - means[i] - min[i]) / (max[i] - min[i])
        else:
            sample[i] = 0
        if sample[i] > 1:
            sample[i] = 1
        if sample[i] < 0:
            sample[i] = 0
    return sample


def get_means_min_max(data):
    # data: 2d array where first dimension is each website, 2nd dimension is the vector for that
    # website
    # returns the mean for each individual variable

    #number of means to get
    num_means = len(data[0])

    #number of samples
    num_samples = len(data)

    # means, min, and max arrays
    mean_totals = [0]*num_means
    min = [1000000000]*num_means
    max = [-100000000]*num_means

    for url_vector in data:
        for i in range(num_means):
            value = url_vector[i]
            if value < min[i]:
                min[i] = value
            if value > max[i]:
                max[i] = value
            mean_totals[i] += value
    means = [x / num_samples for x in mean_totals]
    return means, min, max


def get_incomplete_samples(folder):
    files = os.listdir(folder)
    incomplete_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        if(len(string_attrs) < 11):
            incomplete_data.append(file[:-4])
    return incomplete_data


def get_samples_with_urlvoid_fail(folder, verbose = False):
    files = os.listdir(folder)
    incomplete_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        if(string_attrs[4] == "Nothing found" or string_attrs[4] == "Error" or
                string_attrs[5] == "" or string_attrs[5] == "Error"):
            incomplete_data.append(file[:-4])
    if verbose:
        for sample in incomplete_data:
            print(sample)
    return incomplete_data


def get_samples_with_blacklist_fail(folder, verbose = False):
    files = os.listdir(folder)
    incomplete_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        if(string_attrs[3] == "" or string_attrs[3] == "Error"):
            incomplete_data.append(file[:-4])
    if verbose:
        for sample in incomplete_data:
            print(sample)
    return incomplete_data


def get_samples_with_pagerank_fail(folder, verbose = False):
    files = os.listdir(folder)
    incomplete_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        if(string_attrs[6] == "No pagerank found" or string_attrs[6] == "Error"):
            incomplete_data.append(file[:-4])
    if verbose:
        for sample in incomplete_data:
            print(sample)
    return incomplete_data


def get_samples_with_registration_fail(folder, verbose = False):
    files = os.listdir(folder)
    incomplete_data = []

    for file in files:
        file_path = os.path.join(folder, file)
        string_attrs = extract_attributes_from_file(file_path)
        if(string_attrs[0] == "0000-00-00" or string_attrs[1] == "0000-00-00" or string_attrs[2] == "0000-00-00"):
            incomplete_data.append(file[:-4])
        if (string_attrs[0] == "" or string_attrs[1] == "" or string_attrs[2] == ""):
            incomplete_data.append(file[:-4])

    if verbose:
        for sample in incomplete_data:
            print(sample)
    return incomplete_data


def print_number_of_files(path):
    files = os.listdir(path)
    print(len(files))


def check_legitimate_site_data():
    path = os.path.join(os.getcwd(), "ML", "0")
    data = get_all_preprocessed(path)
    mean, min, max = get_means_min_max(data)
    print(mean)
    print(min)
    print(max)


def get_rid_of_duplicates(folder, txtfile):
    # gets rid of duplicate urls in the text file by checking the urls of the reports in folder
    valid_urls = []

    report_files = os.listdir(folder)
    for sample in report_files:
        samplefile = os.path.join(folder, sample)
        s = open(samplefile, "r+")
        contents = s.readlines()
        url = contents[0]
        url = url[url.find(":") + 2 :]
        valid_urls.append(url)
        s.close()

    # filename must be complete path
    f = open(txtfile, "r+")
    f.seek(0)
    for valid_url in valid_urls:
        f.write(valid_url)
    f.truncate()
    f.close()


def prep_train_val_data():
    #returns a tuple (train, val) where each of those is itself a tuple of the structure
    # (sample data, sample labels), and each of <sample data> and <sample labels> is a
    # pytorch tensor
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    #fill train_data first
    for site_class in [0, 1]:
        path = os.path.join(os.getcwd(), "ML", "Train", str(site_class))
        a = get_all_preprocessed(path)
        train_data.extend(a)
        num_samples = len(a)
        train_labels.extend([site_class]*num_samples)
    b = normalize_samples(train_data)
    c = np.array(b)
    train_tensor_data = torch.from_numpy(c)
    train_tensor_labels = torch.from_numpy(np.array(train_labels))
    train = (train_tensor_data, train_tensor_labels)

    # fill val data
    for site_class in [0, 1]:
        path = os.path.join(os.getcwd(), "ML", "Validation", str(site_class))
        a = get_all_preprocessed(path)
        val_data.extend(a)
        num_samples = len(a)
        val_labels.extend([site_class]*num_samples)
    b = normalize_samples(val_data)
    c = np.array(b)
    val_tensor_data = torch.from_numpy(c)
    val_tensor_labels = torch.from_numpy(np.array(val_labels))
    val = (val_tensor_data, val_tensor_labels)

    return train, val


def create_all_data_csv():
    path = os.path.join(os.getcwd(), "ML")
    csv_filename = os.path.join(path, "all_data.csv")
    f = open(csv_filename, "w")
    f.write("Epiry,Registration,Updated,Blacklisted,ServerLocation,BL_Score,PageRank,"
            "HTTPS,Cert_recieved,Cert_chain,Cert_hostname,Phishing\n")

    raw_data = []
    labels = []

    #make a list of all data vectors and also keep track of labels
    for folder in ["Test", "Train", "Validation"]:
        folder_path = os.path.join(path, folder)
        for site_class in [0, 1]:
            subfolder = os.path.join(folder_path, str(site_class))
            raw_data.extend(get_all_preprocessed(subfolder))
            num_samples = len(os.listdir(subfolder))
            labels.extend([str(site_class)]*num_samples)
    #normalize all samples
    norm_data = normalize_samples(raw_data)
    #write to file
    for i, sample in enumerate(norm_data):
        attribute_string = ""
        for attribute in sample:
            attribute_string += str(attribute)
            attribute_string += ","
        attribute_string += str(labels[i])
        attribute_string += "\n"
        f.write(attribute_string)
    f.close()