from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.document_loaders import Docx2txtLoader
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Dict

# Load environment variables
load_dotenv()

# LLM setup
llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini",
                      api_key=os.getenv("OPENAI_API_KEY"),
                      azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                      api_version=os.getenv("AZURE_API_VERSION"),
                      max_tokens=999)

# 768 dimensions vector embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# MongoDB connection setup
MONGODB_USERNAME = quote_plus(os.getenv("MONGODB_USERNAME"))
MONGODB_PASSWORD = quote_plus(os.getenv("MONGODB_PASSWORD"))
MONGODB_HOST = os.getenv("MONGODB_HOST")
connection_string = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}"
client = MongoClient(connection_string)

# MongoDB Vector store setup
DB_NAME = "aic_mock_prj"
COLLECTION_NAME = "resume_dataset"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "resume_dataset_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
resume_vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# Retriever setup
resume_retriever = resume_vector_store.as_retriever() 

# Chat message history setup
chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=connection_string,
    database_name=DB_NAME,
    collection_name="chat_history",
)

# Resume Chatbot Prompt
RESUME_CHATBOT_PROMPT = """You are an expert resume improvement chatbot with extensive knowledge of industry best practices, ATS optimization, and professional resume writing. Your task is to help improve the user's resume by providing specific, actionable feedback and suggestions. But you should only do that if user ask you to do so.

User's Current Resume:
{user_resume_data}

Previous Conversation History Summary:
{history}

Current User Request:
{question}

Similar Successful Resumes for Reference:
{similar_resumes_content}

Please provide feedback and suggestions focusing on:
1. Content optimization and clarity
2. Professional impact and achievement highlighting
3. Industry-specific keywords and ATS compatibility
4. Format and structure improvements
5. Specific areas mentioned in the user's request

Consider the previous conversation history to:
- Avoid repeating suggestions already discussed
- Build upon previous improvements
- Address any follow-up questions or clarifications
- Maintain context of the user's overall improvement goals

Provide your recommendations in a clear, structured format with:
- Specific suggestions for improvements
- Examples of better phrasing where applicable
- Explanation of why each change would be beneficial
- Priority order of suggested changes

Remember to maintain the user's core experience and qualifications while enhancing their presentation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESUME_CHATBOT_PROMPT),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create a chain that can be invoked multiple times
# chat_chain = RunnableWithMessageHistory(
#     prompt | llm,
#     lambda session_id: MongoDBChatMessageHistory(
#         session_id=session_id,
#         connection_string=connection_string,
#         database_name=DB_NAME,
#         collection_name="chat_history",
#     ),
#     input_messages_key="question",
#     history_messages_key="history",
# )

chat_chain = prompt | llm

# Add this function before the get_chat_response function
def get_chat_history_summary(session_id):
    history = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
        database_name=DB_NAME,
        collection_name="chat_history",
    )
    
    # Get last 5 messages
    messages = history.messages[-8:] if len(history.messages) > 8 else history.messages
    
    if not messages:
        return ""
    
    # Create a summary prompt
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation in no more than 8 sentences, focusing on the main points discussed:"),
        ("human", "{messages}")
    ])
    
    # Format messages into readable text
    formatted_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    # Get summary using the LLM
    summary_chain = summary_prompt | llm
    summary = summary_chain.invoke({"messages": formatted_messages})
    
    return summary.content

def get_chat_response(question):
    # Get chat history summary
    history_summary = get_chat_history_summary("test_session")
    
    # Get similar resumes based on the user's query
    similar_resumes_content = resume_retriever.invoke(question)
    
    return chat_chain.invoke(
        {
            "question": question,
            "user_resume_data": user_resume_data[0].page_content,
            "similar_resumes_content": similar_resumes_content,
            "history": history_summary  # Pass the summary instead of raw history
        },
        config={"configurable": {"session_id": "test_session"}}
    )

# user_resume_data = Docx2txtLoader("data/FPT_CV_DevOpsEngineer_Trieu_Minh_Hieu.docx").load()

# # Now you can use it like:
# response = get_chat_response("How can i improve my resume for AI Engineer position?")
# print(response.content)  # To see the response

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

# Store the current resume path
current_resume_path = None

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global current_resume_path, user_resume_data
    try:
        # Create file path in temp directory
        file_path = f"temp/{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update current resume path
        current_resume_path = file_path
        
        # Load resume data
        user_resume_data = Docx2txtLoader(file_path).load()
        
        return {"message": "Resume uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: Dict):
    try:
        if not current_resume_path:
            raise HTTPException(status_code=400, detail="Please upload a resume first")
        
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Get response from chat chain
        response = get_chat_response(query)
        
        chat_message_history.add_user_message(query)
        chat_message_history.add_ai_message(response.content)
        
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-resume")
async def delete_resume():
    global current_resume_path, user_resume_data
    try:
        if current_resume_path and os.path.exists(current_resume_path):
            os.remove(current_resume_path)
            current_resume_path = None
            user_resume_data = None
        return {"message": "Resume deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)