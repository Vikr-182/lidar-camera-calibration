from chatgptinterface import ChatGPTInteface
import json
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from PIL import Image

def calculate_jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    jaccard_similarity = intersection / union
    return jaccard_similarity

def are_strings_similar(str1, str2, threshold=0.75):
    similarity = calculate_jaccard_similarity(str1, str2)
    return similarity >= threshold

def filter_on_road_vehicles(img, json_list):
    filtered_json = []
    for obj in json_list:
        coords = np.array(obj['bev_centroid']).astype(np.int)*2 + 100
        img = np.array(img)
        if np.all(img[coords[0], coords[1]] == [255, 255, 255]):
            filtered_json.append(obj)
    return filtered_json

def crop_bev(json_list):
    filtered_json = []
    for obj in json_list:
        coords = np.array(obj['bev_centroid'])*2
        if abs(coords[0])<=50 and abs(coords[1])<=50:
            filtered_json.append(obj)
    return filtered_json

def filter_bev(json_list):
    filtered_json = []
    bag_of_words = [
        "car", "truck", "bus", "trailer", "suv", "sedan", "toyota", "mercedes", "bmw", "police", "bike", "cycle", "motorcycle"
    ]
    for obj in json_list:
        flag = False
        for word in bag_of_words:
            if word in obj["llm_message_fixed"].lower():
                flag = True
        if flag:
            filtered_json.append(obj)
    return filtered_json

ORGANIZATION = "org-5ojkqnAHaf8EvPNrTXOWFRgO"
# API_KEY = "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"
API_KEY = "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"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

chatGPTInteface = ChatGPTInteface(API_KEY=API_KEY, organization=ORGANIZATION)
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"
interface = ChatGPTInteface(API_KEY, ORGANIZATION, model_name="gpt-3.5-turbo") # this would become gpt-4
json_list = sorted(os.listdir(save_path))

counts = []
logs = []
for dirind in tqdm(range(0, 100)):
    print(dirind)
    for json_item in json_list:
        if json_item.split("_")[1] == "{0:0=6d}".format(dirind):
            dirs = json_list[dirind]
            print(dirs)
            try:
                gt_json = json.load(open(os.path.join(save_path, dirs, "answer_fixed.json")))
                pred_json = json.load(open(os.path.join(save_path, dirs, "answer_pred.json")))
                gt_bev = Image.open(os.path.join(save_path, dirs, "gt_bev.png"))
                pred_bev = json.load(open(os.path.join(save_path, dirs, "answer_fixed.json")))
            except:
                continue
            for i in gt_json:
                i["llm_message"] = i["llm_message_fixed"]
                # i["tag"] = "".join(i["annotation"]["category_name"][0].split(".")[1:])
                del i["matched_coords"]
                try:
                    del i["bev_coords"]
                except:
                    pass
                del i["llm_message_grit"]
                del i["llm_message_instructblip2"]
                del i["llm_message_fixed"]
                try:
                    del i["llm_message_minigpt4"]
                except:
                    pass
                del i["llm_message_llava"]
                del i["annotation"]

            for i in pred_json:
                del i["matched_coords"]

            question_types = ["SPATIAL_REASONING", "INSTANCE_COUNTING", "VISUAL_REASONING"]
            # gt_json = crop_bev(gt_json)
            # pred_json = crop_bev(pred_json)
            # gt_json = filter_on_road_vehicles(gt_bev, gt_json)
            # pred_json  = filter_on_road_vehicles(gt_bev, pred_json)
            print(len(gt_json), len(pred_json));
            for qntype in question_types:
                response = interface.generate_question(gt_json, question_type=qntype)
                question_with_answer = response["choices"][0]["message"]["content"]
                answer_ind = response["choices"][0]["message"]["content"].lower().find("answer")
                generated_question = question_with_answer[:answer_ind]
                correct_answer = question_with_answer[answer_ind+8:]
                # import pdb; pdb.set_trace()

                # DOING WITH GT ITSELF
                response = interface.generate_conversation(gt_json, generated_question, conversation_type="MCQ")
                
                chatgpt_answer = response["choices"][0]["message"]["content"][3:]
                print("\n\n\n")
                print("selected answer \n", chatgpt_answer)
                # import pdb; pdb.set_trace()
                count = are_strings_similar(chatgpt_answer, correct_answer)
                if not count:
                    if chatgpt_answer in correct_answer or correct_answer in chatgpt_answer:
                        count = True
                print(count)
                counts.append(count)
                logs.append([
                    generated_question,
                    correct_answer,
                    chatgpt_answer
                ])
                # import pdb; pdb.set_trace()

    time.sleep(3)
    # print("\n\n\n\n")
    # print("chatgpt's answer \n", response["choices"][0]["message"]["content"])
    print("Accuracy till here: ", (np.array(counts).sum())/len(counts))

counts = np.array(counts)
print((counts.sum())/len(counts))
df = pd.DataFrame( data=logs, columns=["Question", "Correct Answer", "ChatGPT Answer"])
df.to_csv("logs_gtwgt_100.csv")