from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
from chatbot import chatbot  # Import the chatbot function
import uvicorn
import os
from jose import JWTError, jwt
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


os.environ["TOKENIZERS_PARALLELISM"] = "false"

SECRET_KEY=os.getenv('API_KEY')
ALGORITHM = "HS256"

# Define request and response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# Initialize the FastAPI app
app = FastAPI()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
print(f"Running in {ENVIRONMENT} environment")

if ENVIRONMENT == "prod":
    allowed_origins = ["https://quizgenpt.onrender.com"]
else:
    allowed_origins = ["*"]

print(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_token_from_header(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    token_type, token = auth_header.split(" ")
    if token_type.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    return token

def verify_token(token: str) -> Union[dict, None]:
    try:
        # Decode and verify the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest , token: str = Depends(get_token_from_header)):

 # Verify the JWT token
    payload = verify_token(token)

    question = request.question
    try:
        answer = chatbot(question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a simple root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the chatbot API!"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)