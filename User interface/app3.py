import os
import re
import openai
import pinecone
import requests
import gradio as gr
import torch
from gtts import gTTS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from requests.exceptions import JSONDecodeError
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key
pinecone.init(api_key="5207f7a8-e003-4610-8adb-367ac66812d4", environment='gcp-starter')
index_name = "clinical-bert-index"

# Create a vector database that stores medical knowledge
loader = DirectoryLoader('./medical_data/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split documents into texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize Vectordb
persist_directory = 'db'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

# Create a retrieval QA chain using the vector database as its retriever
retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("For Cuts and Scrapes ")
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Specify the template that the LLM will use to generate its responses
bot_template = '''I want you to act as a medicine advisor for people. 
Explain in simple words how to treat a {medical_complication}'''

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
model_path = "medicalai/ClinicalBERT"
tokenizer_str = tokenizer.__class__.__name__ 

# Create Prompt
prompt = PromptTemplate(
    input_variables=['medical_complication'],
    template=bot_template 
)

# Specify the LLM that you want to use as the language model
llm = OpenAI(temperature=0.8)
chain1 = LLMChain(llm=llm, prompt=prompt)

# Create the retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Global variables
global_filepath = None
global_feedback = None
chatgpt_response = ""
modMed_response = ""
trigger_words = ""


def preprocess_text(text):
    # Preprocess the input text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    embeddings_list = embeddings.squeeze().tolist() 

    data_to_insert = [("id_{}".format(i), vec) for i, vec in enumerate(embeddings_list)]

    return data_to_insert[1][1]


# API key for OpenAI
my_key = os.getenv("OPENAI_API_KEY")
openai.api_key = my_key

# Initialize Pinecone index
pinecone_index = pinecone.Index(index_name=index_name)

# Define function to retrieve embeddings from Pinecone
def retrieve_embeddings_from_pinecone(query):
    response = pinecone_index.query(
        vector=query,
        top_k=3,
        include_values=True
    )
    

    # Assuming 'response' is your Pinecone query response
    matches = response.get("matches", [])

    # Extracting only the "values" field from each match
    values_list = [match.get("values", []) for match in matches]

    # Printing the result
    return values_list


# Function to process user input
def process_user_input(audio_filepath, feedback):
    global global_filepath
    audio = open(audio_filepath, "rb")
    global_filepath = audio_filepath

    transcript = openai.Audio.transcribe("whisper-1", audio)

    return transcript["text"]


# Function to find trigger words
def findTriggerWords(user_input):
    prompt = (
        f"Given this user input: {user_input}\n"
        "Task: Identify and return important keywords from the user input. "
        "These keywords are crucial for understanding the user's intent and finding a relevant solution. "
        "Consider context and relevance. Provide a numbered list up to 5 keywords or less"
    )

    response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=prompt,
        max_tokens=500, 
        temperature=0.7,  
    )

    ChatGPT_response = response['choices'][0]['text']
    return ChatGPT_response.replace(".", "").replace("\n", "", 1).strip()


# Function to make an API call
def api_call(url):
    try:
        response = requests.post(url)
        if response.status_code == 200:
            updated_data = response.json()
            print(f"Updated Database: {updated_data}")
            return updated_data
        else:
            print(f"Error updating database: {response.status_code}")
            print(response.text)
            return None
    except JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Response text: {response.text}")
        return None


# Function to process feedback
def process_feedback(feedback, current_filepath):
    global global_filepath, global_feedback, chatgpt_response, modMed_response
    ans = ""
    url = ""

    if feedback in ["🏥 ModMed", "🤖 ChatGPT"] and global_feedback == None:
        global_feedback = feedback
        incr_query_string = ''.join(char for char in feedback if char.isalnum())
        url = f"http://localhost:5000/increment_likes/{incr_query_string}"
        print("new audio file")

    elif feedback in ["🏥 ModMed", "🤖 ChatGPT"] and global_feedback != None:
        print("same audio file, different radio button")
        global_feedback = feedback
        decr_query_string = ''.join(char for char in global_feedback if char.isalnum())
        incr_query_string = ''.join(char for char in feedback if char.isalnum())
        decrement_url = f"http://localhost:5000/decrement_likes/{decr_query_string}"
        increment_url = f"http://localhost:5000/increment_likes/{incr_query_string}"

        # Decrement likes
        decrement_data = api_call(decrement_url)

        if decrement_data:
            # Increment likes if decrement was successful
            url = increment_url

    else:
        return ans

    updated_data = api_call(url)

    if updated_data:
        if feedback == "🏥 ModMed":
            chatgpt_response = ""
            modMed_response = "True"
        elif feedback == "🤖 ChatGPT":
            modMed_response = ""
            chatgpt_response = "True"

        preferred_strings = ", ".join(string for string in ["ModMed", "ChatGPT"] if string != incr_query_string)
        ans = f"{updated_data['Likes']}/{updated_data['TotalLikes']} People preferred {incr_query_string} over {preferred_strings}.\nThank you! 👍"

    return ans


# Function to handle the chatbot logic
def chatbot(microphone_filepath, upload_filepath, feedback):
    global global_filepath, global_feedback, chatgpt_response, modMed_response, trigger_words
    print("Feedback", feedback)
    
    if microphone_filepath is not None:
        audio_filepath = microphone_filepath
    elif upload_filepath is not None:
        audio_filepath = upload_filepath
    else:
        global_filepath = global_feedback = None
        chatgpt_response = ""
        modMed_response = ""
        trigger_words = ""
        print(trigger_words)
        global_filepath = None
        global_feedback = None
        return None, None, None, None, None        

    # Process user input
    if global_filepath != audio_filepath:
        user_input = process_user_input(audio_filepath, feedback)
        trigger_words = findTriggerWords(user_input)
    elif feedback == "Clear" and global_filepath != None:
        feedback = ""
        chatgpt_response = ""
        modMed_response = ""
        trigger_words = ""
        global_filepath = None
        global_feedback = None
        return None, None, None, None, None
    else:
        user_input = None

    if user_input is not None or feedback != global_feedback:
        # Get the chatbot response
        chatgpt_prompt = f"Act like a medical bot and return at most 5 sentences if the user_input isn't a medical question then answer the question in general: user_input:\n{user_input}"
        llm_response = qa_chain(chatgpt_prompt)
        prompt_response = chain1(user_input)

        r_embeddings = retrieve_embeddings_from_pinecone(preprocess_text(user_input))
        print(r_embeddings)
        
        f_modMed_response, f_chatgpt_response = process_llm_response(llm_response, prompt_response)
        ans = process_feedback(feedback, global_filepath)

        if modMed_response  == "" and chatgpt_response != "": 
            print("CHATGPT FEEDBACK")
            clean_response = f_chatgpt_response.split('<br>')[0]
            audio_response = text_to_speech(clean_response) 
            return gr.make_waveform(audio_response, animate=True), None, f_chatgpt_response, trigger_words, ans 

        elif modMed_response  != "" and chatgpt_response == "": 
            print("MODMED FEEDBACK")
            clean_response = f_modMed_response.split('<br>')[0]
            audio_response = text_to_speech(clean_response) 
            return gr.make_waveform(audio_response, animate=True), f_modMed_response, None, trigger_words, ans 
        else:
            print("NO FEEDBACK")
            audio_response = text_to_speech(f_modMed_response.split('<br>')[0])
            return gr.make_waveform(audio_response, animate=True), f_modMed_response, f_chatgpt_response, trigger_words, ans
    
    return None, None, None, None, None


def process_llm_response(llm_response, prompt_response):
    ChatGPT_response = llm_response['result']
    ModMed_response = str(prompt_response["text"])

    ChatGPT_image_html = f'<img src="https://static.vecteezy.com/system/resources/previews/021/495/996/original/chatgpt-openai-logo-icon-free-png.png" alt="image" style="width:25px;height:25px;display:inline-block;">'
    ModMed_image_html = f'<img src="https://png.pngtree.com/png-clipart/20230707/original/pngtree-green-approved-stamp-with-check-mark-symbol-vector-png-image_9271227.png" alt="image" style="width:25px;height:25px;display:inline-block;">'
    
    ModMed_source = f'<br><span style="color: darkgray;">ModMedicine Certified {ModMed_image_html}</span>'
    ChatGPT_source = f'<br><span style="color: darkgray;">ChatGPT {ChatGPT_image_html}</span>'

    return (
        ModMed_response + ModMed_source,
        ChatGPT_response + ChatGPT_source
    )


def play_response(response=None):
    if response is not None:
        audio_path = text_to_speech(response)
        return gr.Audio(audio_path)
    else:
        return None


def text_to_speech(text):
    # Find the index of 'ModMed Certified' (case insensitive)
    certified_index = text.lower().find('ModMedicine certified')
    chatgpt_index = text.lower().find('ChatGPT')
    
    if certified_index != -1 or chatgpt_index != -1:
        # Cut off the text after 'ModMed Certified'
        text = text[:certified_index]

    tts = gTTS(text=text, lang='en', tld='co.uk')
    audio_path = 'response.mp3'
    tts.save(audio_path)
    
    return audio_path


# Feedback radio button choices
feedback_buttons = gr.Radio(
    choices=["🏥 ModMed", "🤖 ChatGPT", "Clear"],
    label="Which solution was better?",
    default=None  # Set the default value to None
)

# Gradio Interface
demo = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Audio(source="microphone", type="filepath"),
        gr.Audio(source="upload", type="filepath"),
        feedback_buttons
    ],
    outputs=[
        gr.Video(autoplay=True, label="ModMedicine"),
        gr.outputs.HTML(label="ModMed Response"),
        gr.outputs.HTML(label="ChatGpt Response"),
        gr.Text(label="Trigger words"),
        gr.Text(label="Feedback")
    ],
    examples=[
        ["./dummy_audio1.mp3"],
        ["./dummy_audio2.mp3"]
    ],
    title="First-Aid Bot",
    description='<img src="https://drive.google.com/uc?id=1fWN0xn_KXLb0fCtTAyoxwacwzyH6w4am&export=download" alt="logo" style="display: block; margin: auto; width:125px;height:125px;">',
    live=True
)

# Launch the Gradio interface
demo.launch(server_name="0.0.0.0", server_port=7860)

