import gzip
import json
import random
import re

import pandas as pd


def removeLastN(S, N):
    S = S[:len(S) - N]
    return S


def parse_dataset(file_path):
    pattern = re.compile(r"^\/a\/\[.*\/,\/c\/en\/.*\/,\/c\/en\/")
    save_dict = {}

    with gzip.open(file_path, 'r') as tar:
        for line in tar:
            string = line.decode('utf-8')
            if pattern.search(string):
                my_str = string.split("\t")
                if 'surfaceText' in my_str[4]:
                    my_json = json.loads(my_str[4])
                    # single_str.write(my_json['surfaceText']+"\n")
                    if my_json['surfaceStart'] not in save_dict:
                        save_dict[my_json['surfaceStart']] = [my_json['surfaceText'].replace('[', '').replace(']', '')]
                    else:
                        save_dict[my_json['surfaceStart']].append(
                            my_json['surfaceText'].replace('[', '').replace(']', ''))
    return save_dict


def sample_concatenate_triplets(data, file):
    for key, arr in data.items():
        sub_str = ""
        for sentence in (random.sample(arr, 5) if (len(arr) > 5) else arr):
            sub_str += sentence.replace('[', '').replace(']', '') + " && "
        file.write(str(removeLastN(sub_str, 4)) + "\n")


def mask_words(file_path):
    text = []
    target = []
    input = []

    with open(file_path) as txt_file:
        for line in txt_file:
            text.append(line[:-1])
            strings = line.split(" && ")
            input_str = ""
            target_str = ""
            for string in strings:
                input_ind = 0
                target_ind = 0
                new_str = ""
                tempt_target_str = ""
                words = list(map(str, string.split()))
                all_words = words.copy()
                word1 = random.choice(words)
                words.remove(word1)
                word2 = random.choice(words)
                for word in all_words:
                    if word == word1 or word == word2:
                        new_str += f"<extra_id_{input_ind}> "
                        tempt_target_str += word + " "
                        input_ind += 1
                    else:
                        new_str += word + " "
                        tempt_target_str += f"<extra_id_{target_ind}> "
                        target_ind += 1
                input_str += new_str + "&& "
                target_str += tempt_target_str + "&& "

            input.append(removeLastN(input_str, 4))
            target.append(removeLastN(target_str, 4))

    return input, target, text


if __name__ == "__main__":
    print("Start Concetpnet preparation")
    conceptnet_path = 'data/conceptnet-assertions-5.7.0.csv.gz'

    # Parse ConceptNet triplets and save only English entries as {"Subject": ["SurfaceText1", "SurfaceText2", ...]}
    data_triplets = parse_dataset(conceptnet_path)
    with open("data/en_conceptnet_surfacetext.json", "w") as fp:
        json.dump(data_triplets, fp)

    with open('data/en_conceptnet_surfacetext.json') as json_file:
        data = json.load(json_file)

    # Randomly sample 5 triplets from each Subject and concatenate them into one string
    f = open("data/sampled_5triplets_to_one_string.txt", "w")
    sample_concatenate_triplets(data, f)

    # Randomly mask 2 words in each SurfaceText
    sampled_triplets_path = "data/sampled_5triplets_to_one_string.txt"
    input, target, text = mask_words(sampled_triplets_path)

    df = pd.DataFrame.from_dict({
        'input': input,
        'target': target,
        'text': text
    })

    df.to_csv('data/input_target_one_string.csv')
    print("FINISHED")

