from chatgptinterface import ChatGPTInteface
import json

ORGANIZATION = "org-5ojkqnAHaf8EvPNrTXOWFRgO"
API_KEY = "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"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

chatGPTInteface = ChatGPTInteface(API_KEY=API_KEY, organization=ORGANIZATION)

data = [
    {
        "object_id": 1,
        "bev_centroid": [-39.0, 0.0],
        "bev_area": 45,
        "llm_message": "The object is a long, straight road with a yellow line down the center. There are no other visible objects. The road appears to be in a rural area with fields on either side and no other buildings or roads in sight. There is a blue sky in the background, with some clouds visible. The lighting in the scene is bright and sunny. The perspective is looking down the road, with the horizon in the distance."
    },
    {
        "object_id": 2,
        "bev_centroid": [-10.0, 7.5],
        "bev_area": 45,
        "llm_message": "This is a black car with its front facing towards the camera. The car has a sleek design, with a sharply angled front end and narrow, straight headlights. The car's body is also angled, with the rear end pointing towards the camera."
    },
    {
        "object_id": 3,
        "bev_centroid": [-50.0, -50.0],
        "bev_area": 0,
        "llm_message": "The object is a road that runs horizontally through the frame. It appears to be a single lane road with no other cars or pedestrians in sight. The sky is clear, with a few clouds in the distance. The road is straight and flat, and there is no obstruction or obstruction in the way. The asphalt is shiny and smooth."
    }
]

interface = ChatGPTInteface(API_KEY, ORGANIZATION, model_name="gpt-3.5-turbo")
# MCQ example
response = interface.generate_question(data, question_type="VISUAL_REASONING")
question_with_answer = response["choices"][0]["message"]["content"]
answer_ind = response["choices"][0]["message"]["content"].lower().find("answer")
generated_question = question_with_answer[:answer_ind]
correct_answer = question_with_answer[answer_ind+8:]
print(generated_question)
print(correct_answer)
response = interface.generate_conversation(data, generated_question, conversation_type="MCQ")
print(response)

# Spatial example
response = interface.generate_conversation(data, "Describe the objects in front of me.", conversation_type="SPATIAL")
print(json.loads(response["choices"][0]["message"]["content"]))

response = interface.generate_conversation(data, "Describe the objects within 100m of me.", conversation_type="SPATIAL")
print(json.loads(response["choices"][0]["message"]["content"]))

