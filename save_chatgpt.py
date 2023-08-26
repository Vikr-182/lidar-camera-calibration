from chatgptinterface import ChatGPTInteface
import json

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

data = [
    {
        "object_id": 1,
        "bev_centroid": [-39.0, 0.0],
        "bev_area": 45,
        "tag": "truck",
        "llm_message": "The object is a long, straight road with a yellow line down the center. There are no other visible objects. The road appears to be in a rural area with fields on either side and no other buildings or roads in sight. There is a blue sky in the background, with some clouds visible. The lighting in the scene is bright and sunny. The perspective is looking down the road, with the horizon in the distance."
    },
    {
        "object_id": 2,
        "bev_centroid": [-10.0, 7.5],
        "bev_area": 45,
        "tag": "car",
        "llm_message": "This is a black car with its front facing towards the camera. The car has a sleek design, with a sharply angled front end and narrow, straight headlights. The car's body is also angled, with the rear end pointing towards the camera."
    },
    {
        "object_id": 3,
        "bev_centroid": [-50.0, -50.0],
        "bev_area": 0,
        "tag": "car",        
        "llm_message": "The object is a road that runs horizontally through the frame. It appears to be a single lane road with no other cars or pedestrians in sight. The sky is clear, with a few clouds in the distance. The road is straight and flat, and there is no obstruction or obstruction in the way. The asphalt is shiny and smooth."
    }
]

interface = ChatGPTInteface(API_KEY, ORGANIZATION, model_name="gpt-3.5-turbo")
# MCQ example
# response = interface.generate_question(data, question_type="VISUAL_REASONING")
# question_with_answer = response["choices"][0]["message"]["content"]
# print("response \n", question_with_answer)
# print("\n\n")
# answer_ind = response["choices"][0]["message"]["content"].lower().find("answer")
# generated_question = question_with_answer[:answer_ind]
# correct_answer = question_with_answer[answer_ind+8:]
# print("separated question \n", generated_question)
# print("\n\n")
# print("separated answer \n", correct_answer)



# response = interface.generate_conversation(data, generated_question, conversation_type="MCQ")
# chatgpt_answer = response["choices"][0]["message"]["content"]

# print("\n\n\n\n")
# print("chatgpt's answer \n", response["choices"][0]["message"]["content"])

# count = are_strings_similar(chatgpt_answer, correct_answer)
# print(count)

# Spatial example
# response = interface.generate_conversation(data, "Find all the objects which have tag as car as are within 100m to ego-vehicle. Out of this list, then find the closest object to ego-vehicle.", conversation_type="SPATIAL")
# print(json.loads(response["choices"][0]["message"]["content"]))

response = interface.generate_conversation(data, "Given a list_of_objects, find the objects within 20m which are between 5-10m in size to ego-vehicle.", conversation_type="SPATIAL")
print(json.loads(response["choices"][0]["message"]["content"]))


# response = interface.generate_conversation(data, "Filter the objects within 100m of ego-vehicle.", conversation_type="SPATIAL")
# print(json.loads(response["choices"][0]["message"]["content"]))

