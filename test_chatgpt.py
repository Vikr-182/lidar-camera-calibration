from chatgptinterface import ChatGPTInteface
import json
import numpy as np
import pandas as pd
import os

def calculate_jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    jaccard_similarity = intersection / union
    return jaccard_similarity

def are_strings_similar(str1, str2, threshold=0.9):
    similarity = calculate_jaccard_similarity(str1, str2)
    return similarity >= threshold

ORGANIZATION = "org-5ojkqnAHaf8EvPNrTXOWFRgO"
API_KEY = "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"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

chatGPTInteface = ChatGPTInteface(API_KEY=API_KEY, organization=ORGANIZATION)
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"
interface = ChatGPTInteface(API_KEY, ORGANIZATION, model_name="gpt-3.5-turbo") # this would become gpt-4
json_list = sorted(os.listdir(save_path))

counts = []
logs = []
for dirind in range(0, 3):
    print(dirind)
    for json_item in json_list:
        if json_item.split("_")[1] == "{0:0=6d}".format(dirind):
            dirs = json_list[dirind]
            print(dirs)
            gt_json = json.load(open(os.path.join(save_path, dirs, "answer_fixed.json")))
            pred_json = json.load(open(os.path.join(save_path, dirs, "answer_pred.json")))
            print(gt_json[0].keys())
            for i in gt_json:
                del i["matched_coords"]
                del i["llm_message_grit"]
                del i["llm_message_instructblip2"]
                del i["llm_message"]
                del i["llm_message_llava"]
                del i["annotation"]

            for i in pred_json:
                del i["matched_coords"]

            question_types = ["SPATIAL_REASONING", "INSTANCE_ATTRIBUTE", "INSTANCE_COUNTING", "VISUAL_REASONING"]
            for qntype in question_types:
                response = interface.generate_question(gt_json, question_type=qntype)
                question_with_answer = response["choices"][0]["message"]["content"]
                print("response \n", question_with_answer)
                print("\n\n")
                answer_ind = response["choices"][0]["message"]["content"].lower().find("answer")
                generated_question = question_with_answer[:answer_ind]
                correct_answer = question_with_answer[answer_ind+8:]
                print("separated question \n", generated_question)
                print("\n\n")
                print("separated answer \n", correct_answer)
                response = interface.generate_conversation(pred_json, generated_question, conversation_type="MCQ")
                chatgpt_answer = response["choices"][0]["message"]["content"]
                print("\n\n\n")
                print("selected answer \n", chatgpt_answer)
                count = are_strings_similar(chatgpt_answer, correct_answer)
                print(count)
                counts.append(count)
                logs.append([
                    generated_question,
                    correct_answer,
                    chatgpt_answer
                ])
                import pdb; pdb.set_trace()
    # print("\n\n\n\n")
    # print("chatgpt's answer \n", response["choices"][0]["message"]["content"])

counts = np.array(counts)
print((counts.sum())/len(counts))
df = pd.DataFrame( data=logs, columns=["Question", "Correct Answer", "ChatGPT Answer"])
df.to_csv("logs.csv")