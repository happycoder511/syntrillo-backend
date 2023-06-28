import os

from flask import Flask, request
from flask_cors import CORS

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

import openai

### Load environment variables from .env
load_dotenv()

### Directory path for uploading files and the extension of allowed files.
UPLOAD_FOLDER = './traindata'
ALLOWED_EXTENSIONS = {'csv'}

### Create Flask APP
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ### Load already trainned data
# if os.path.exists("./store/index.faiss"):
#     docsearch = FAISS.load_local("./store", OpenAIEmbeddings())
# else:
#     docsearch = FAISS.from_documents([Document(page_content="This is ChatBot.\n\n")], OpenAIEmbeddings())

# ### Check the file allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ### Create conversational Retrieval Chain
# chain = ConversationalRetrievalChain.from_llm(
#     llm = ChatOpenAI(
#         temperature=0.1,
#         model_name=os.getenv('OPENAI_MODEL'),
#         openai_api_key=os.getenv('OPENAI_API_KEY'),
#         request_timeout=120
#     ),
#     retriever=docsearch.as_retriever()
# )

# ### File Upload
# @app.route('/api/upload', methods=['POST'])
# def upload():
#     global reader, raw_text, texts, embeddings, docsearch, chain

#     ### Check the file exist and allowed
#     if 'file' not in request.files:
#         return {"state": "error", "message": "No file part"}
#     file = request.files['file']
#     if file.filename == '':
#         return {"state": "error", "message": "No selected file"}
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         loader = CSVLoader(os.path.join(app.config['UPLOAD_FOLDER'], filename), encoding='cp1252')
#         data = loader.load()

#         ### Enbed the data
#         embeddings = OpenAIEmbeddings()
#         vectors = FAISS.from_documents(data, embeddings)

#         ### Merge the new data into local data 
#         if os.path.exists("./store/index.faiss"):
#             docsearch = FAISS.load_local("./store", OpenAIEmbeddings())
#             docsearch.merge_from(vectors)
#         else:
#             docsearch = vectors
#         docsearch.save_local("./store")

#         ### Update conversational retrieval chain
#         chain = ConversationalRetrievalChain.from_llm(
#             llm = ChatOpenAI(
#                 temperature=0.1,
#                 model_name=os.getenv('OPENAI_MODEL'),
#                 openai_api_key=os.getenv('OPENAI_API_KEY'),
#                 request_timeout=120
#             ),
#             retriever=docsearch.as_retriever()
#         )

#         return {"state": "success"}
#     return {"state": "error", "message": "Invalid file format"}

openai.api_key = "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"

### Chat
@app.route('/api/chat', methods=['POST'])
def chat():
    print("ok")
    print(request.json)
    query = request.json['prompt']
    # completion = chain({"question": query, "chat_history": []})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant helping stroke survivors only. Here is some data. You can refer to this."}, {"role": "user", "content": query}],
        # max_tokens=2000
    )
    print(response.choices[0].message.content)
    return {"answer": response.choices[0].message.content}

if __name__ == '__main__':
    app.run()