import os
import openai

scene_data_format = {
	"object_id": "a unique object id",
	"bev_centroid": "2D coordinates of the center of the object",
	"bev_area": "Size of the 2D object",    
	"llm_message": "a brief caption for the object"
}

answer_format = {
   "inferred_query": "your interpretation of the user query in a succinct form",
   "relevant_objects": "python list of all relevant object ids for the user query",
   "query_achievable": "whether or not the user-specified query is achievable using the objects and descriptions provided in the scene.",
   "spatial_reasoning_functions": "If the query needs calling one or more spatial reasoning functions, this field contains a list of function calls that conform to the API above. Else, this field contains an empty list.",
   "explanation": "A brief explanation of what the relevant objects, and how it addresses the task."
}

spatial_operators_list = [
	"filter_front() : finds the objects to front of the ego-vehicle",
	"filter_left() : finds the objects to left of the ego-vehicle",
	"filter_right() : finds the objects to right of the ego-vehicle",
	"filter_rear() : finds the objects to rear of the ego-vehicle",
	"find_distance(object_id_1, object_id_2) : finds the distance within 2 objects object_id_1 and object_id_2",
	"find_objects_within_distance(object_id, distance) : finds the objects within distance to the object with id object_id",
    "get_k_closest_objects(object_id, k) : get the k closest objects to the object with id object_id"
]

def add_question_context(question_type):
    if question_type == "SPATIAL_REASONING":
        return "The question should be about spatial relations between two objects. The question should be mainly based on the coordinates of the two objects. To answer the question, one should find the two mentioned objects, and find their relative spatial relation to answer the question."
    elif question_type == "INSTANCE_ATTRIBUTE":
        return "The question should be about the attribute of a certain object, such as its color, shape or fine-grained type."
    elif question_type == "INSTANCE_COUNTING":
        return "The question should involve the number of appearance of a certain object. Start with 'How many ....'. The choices of the question should be numbers. To answer the question, one should find and count all of the mentioned objects in the image."
    elif question_type == "VISUAL_REASONING":
        return "Create complex question beyond describing the scene. To answer such question, one should first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first."
    else:
        return ""

def setup_question_generation(question_type):
    message = f"You will be given, as input a 2D road scene in Bird's Eye View, as a list. The ego-vehicle is at (0,0) facing along the positive Y-axis. Each entry in the list describes one object in the scene, with the following five fields: \
                    \n\n {str(scene_data_format)}  \n\n  \
                Once you have parsed the JSON and are ready to generate question about the scene, Create a multi-choice question about the image, and provide the choices and answer. Each question should have 4 options, out of which only 1 should be correct."
    message += add_question_context(question_type)
    return {
        "role": "system",
        "content": message
    }

def add_conversation_context(conversation_type):
    if conversation_type == "MCQ":
        return  "The user will then begin to ask Multiple Choice Questions, and the task is to answer various user queries about the scene. For each question, answer with just one correct option. You should only output the correct answer. Do not give any explanation"

    if conversation_type == "SPATIAL":
        return  f"The user will then begin to ask questions, and the task is to answer various user queries about the scene. These questions will involve spatial reasoning. To assist with such queries, we have the following available functions:.\
                    {str(spatial_operators_list)}\
                    For each user question, respond with a JSON dictionary with the following fields:\
                    {str(answer_format)}\
                    Only output this JSON. Do not output any explanation."

def setup_conversation(conversation_type="MCQ"):
    message = f"You will be given, as input a 2D road scene in Bird's Eye View, as a list. The ego-vehicle is at (0,0) facing along the positive Y-axis. Each entry in the list describes one object in the scene, with the following five fields: \
                    \n\n {str(scene_data_format)}  \n\n  \
                Once you have parsed the JSON and are ready to answer questions about the scene, please wait for the user to input JSON."
    message += add_conversation_context(conversation_type)
    return {
        "role": "system",
        "content": message
    }

class ChatGPTInteface:
    def __init__(self, API_KEY, organization, model_name="gpt-3.5-turbo") -> None:
        openai.api_key = API_KEY
        openai.organization = organization
        self.model_name = model_name
        
    def generate_question(self, data, question_type="DEFAULT"):
        system_message = setup_question_generation(question_type)
        user_message = {
            "role": "user",
            "content": str(data)
        }
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                system_message,
                user_message
            ],
            temperature=0,
            max_tokens=1024
        )
        return response

    def generate_conversation(self, data, question, conversation_type):
        system_message = setup_conversation(conversation_type)
        user_message = {
            "role": "user",
            "content": str(data) + ". The question is as follows \n" + str(question)
        }
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                system_message,
                user_message
            ],
            temperature=0,
            max_tokens=1024
        )
        return response


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