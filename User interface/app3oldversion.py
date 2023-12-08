import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import openai
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import numpy as np
import umap
from transformers import AutoTokenizer, AutoModel
import torch
import re
from gtts import gTTS
import time
from colorama import Fore, Style
import pinecone

load_dotenv()

pinecone.init(api_key="5207f7a8-e003-4610-8adb-367ac66812d4", environment='gcp-starter')
index_name = "clinical-bert-index"
# Create a vector database that stores the medical knowledge.
loader = DirectoryLoader('./medical_data/', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

len(texts)

texts[3]

persist_directory = 'db'
embedding = OpenAIEmbeddings()

# Initialize Vectordb
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
vectordb.persist()
# Create a retrieval QA chain that uses the vector database as its retriever.
retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("For Cuts and Scrapes ")
len(docs)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
retriever.search_type
retriever.search_kwargs
# Specify the template that the LLM will use to generate its responses.
bot_template = '''I want you to act as a medicine advisor for people. 
Explain in simple words how to treat a {medical_complication}'''

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
model_path = "medicalai/ClinicalBERT"
tokenizer_str = tokenizer.__class__.__name__ 
# Create Prompt
prompt=PromptTemplate(
    input_variables=['medical_complication'],
    template=bot_template 
)

# Specify the LLM that you want to use as the language model.
llm = OpenAI(temperature=0.8)
chain1=LLMChain(llm=llm,prompt=prompt)

# Create the retrieval QA chain.
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, 
return_source_documents=True)

def preprocess_text(text):

    text = text.lower()
    # Removing special characters, punctuation, and extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generating the embeddings using the ClinicalBERT model
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    # Convert the embeddings tensor to a list and remove the batch dimension
    embeddings_list = embeddings.squeeze().tolist()  # Converts the tensor to a list of 433 vectors, each of size 768
    # Convert the embeddings tensor to a numpy array
    embeddings_array = embeddings.squeeze().numpy()  # This will be a numpy array of shape (num_tokens, 768)

    # Perform dimensionality reduction using UMAP to reduce to 768 dimensions
    reducer = umap.UMAP(n_components=768)
    reduced_embeddings = reducer.fit_transform(embeddings_array)


    return reduced_embeddings

pinecone_index = pinecone.Index(index_name=index_name)
# pinecone_index.upsert(vectors=data_to_insert)s

def retrieve_embeddings_from_pinecone(query):
    # Perform a nearest neighbor search to retrieve embeddings based on the query
    results = pinecone_index.query(
    vector=query,
    top_k=3,
    include_values=True
    )
    # Extract the retrieved embeddings from the results
    retrieved_embeddings = results[0].vectors
    return retrieved_embeddings
chat_history = []

def chatbot(filepath, feedback):
    audio = open(filepath, "rb")

    my_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = my_key
    transcript = openai.Audio.transcribe("whisper-1", audio)

    # Get the current time in seconds
    current_time = time.time()

    # Store user input in the chat history with formatted timestamp
    user_input = transcript["text"]
    user_timestamp = round(current_time - time.time(), 1)  # Time in seconds
    chat_history.append((f"User: {user_input} {user_timestamp}s\n", "green"))

    #retrieved_embeddings = retrieve_embeddings_from_pinecone(preprocess_text(user_input))
    # Get the chatbot response
    llm_response = qa_chain(user_input)

    prompt_response = chain1(user_input)
    
    text_response = process_llm_response(llm_response, prompt_response)
    audio_response = text_to_speech(text_response)
    gif_path = 'waveform.gif'  # Provide the path to your GIF file

    # Extract the first sentence from the bot response
    first_sentence = text_response.split('.')[0]

    # Store chatbot response in the chat history with formatted timestamp
    bot_timestamp = round(time.time() - current_time, 1)  # Time in seconds
    chat_history.append((f"Bot: {first_sentence} {bot_timestamp}s\n", "red"))

    # Join the chat history into a single string
    chat_history_str = chat_history

    ans = ""
    # Process user feedback
    if feedback == "👍 Thumbs Up":
        # Positive feedback response
        ans = "Thanks for the positive feedback, have a great day!"
    elif feedback == "👎 Thumbs Down":
        # Negative feedback response
        ans = "Sorry about your unsatisfying experience with our software, we will work on improving soon!"

    return audio_response, gif_path, text_response, chat_history_str, ans, prompt_response

def process_llm_response(llm_response, prompt_response):
    responses = llm_response['result']
    prompt_ans = str(prompt_response["text"])
    
    Modimage_html = f'<img src="https://png.pngtree.com/png-clipart/20230707/original/pngtree-green-approved-stamp-with-check-mark-symbol-vector-png-image_9271227.png" alt="image" style="width:25px;height:25px;display:inline-block;">'
    GPTimage_html = f'<img src="https://static.vecteezy.com/system/resources/previews/021/495/996/original/chatgpt-openai-logo-icon-free-png.png" alt="image" style="width:25px;height:25px;display:inline-block;">'
    
    return 'ModMedicine: ' + prompt_ans + Modimage_html + '\n\nChatGPT '+ responses + GPTimage_html
    
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

feedback_buttons = gr.Radio(choices=["👍 Thumbs Up", "👎 Thumbs Down"], label="Feedback")

demo = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Audio(source="microphone", type="filepath"), feedback_buttons
    ],
    outputs=[
        gr.Audio(play_response),
        gr.Image(type="filepath"),
        gr.outputs.HTML(label="Response"),
        gr.HighlightedText(label="Chat History", default="", color_map={"green": "green", "red": "red"}),
        gr.Text(label="Response")
    ],
    title="First-Aid Bot",
    description='<img src="https://drive.google.com/uc?id=1fWN0xn_KXLb0fCtTAyoxwacwzyH6w4am&export=download" alt="logo" style="display: block; margin: auto; width:125px;height:125px;">',
    live=True
)
# Launch the Gradio interface
demo.launch(server_name="0.0.0.0", server_port=7860) #If 0.0.0.0 doesn't work run 127.0.0.1:7860