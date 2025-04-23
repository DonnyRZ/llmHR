# --- Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import functions from our core modules ---
from core.rag_service import retrieve_context, format_context_for_llm
from core.llm_service import get_ollama_response, analyze_query_intent


# --- Pydantic Model for Request Body ---
class ChatRequest(BaseModel):
    message: str


# --- Create FastAPI App Instance ---
app = FastAPI(
    title="Resume Insight Assistant API",
    description="API for the HR Chatbot using Llama 3.2",
    version="0.1.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "http://192.168.99.152:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Configuration ---


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Resume Insight Assistant Backend!"}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to receive user messages, analyze intent/criteria, retrieve context,
    (potentially bypass LLM if no context), construct prompt,
    get an LLM response, and return it.
    """
    user_message = request.message
    print(f"\nReceived user message via API: '{user_message}'")

    # --- Analyze the query first ---
    analyzed_query = analyze_query_intent(user_message)
    print(f"--- Analyzed Query Results: {analyzed_query} ---")

    # 1. Retrieve Context using rag_service
    relevant_candidates = retrieve_context(query=user_message, analyzed_query=analyzed_query)

    # --- Step 12 Check: Check if context was found ---
    if not relevant_candidates:
        print("--- No relevant context found by retrieve_context. Bypassing LLM. ---")
        not_found_message = "I couldn't find any candidate information relevant to your query in the current dataset. Could you please try rephrasing?"
        return {"reply": not_found_message}
    else:
        print(f"--- Found {len(relevant_candidates)} relevant candidate(s). Proceeding with LLM. ---")

        # 2. Format Context for LLM using rag_service
        formatted_context = format_context_for_llm(relevant_candidates)
        print(f"--- Formatted Context for LLM ---")
        print(formatted_context)
        print(f"--- End Formatted Context ---")

        # --- Prompt Tuning Iteration 2 (Added Few-Shot Example) ---
        # 3. Construct the Prompt with refined instructions and an example
        prompt = f"""You are an HR assistant chatbot. Your task is to answer the user's question based *strictly* and *only* on the provided context about candidates.

        **Context:**
        ---
        {formatted_context}
        ---

        **User Question:** {user_message}

        **Instructions:**
        1. Examine the provided 'Context' carefully.
        2. Answer the 'User Question' using *only* information found within the 'Context'. **List *all* relevant candidates or details found.**
        3. Do not add any information that is not explicitly stated in the 'Context'. Do not make assumptions or use external knowledge.
        4. If the 'Context' does not contain the information needed to answer the question, you MUST respond with "I cannot answer the question based on the provided candidate information." or a very similar phrase. Do not attempt to answer anyway.
        # Instruction 5 (Conciseness) removed previously

        **Example Interaction:**
        --- Example Start ---
        Context:
        - Name: Frank
          Skills: Java, Spring
        - Name: Grace
          Skills: Java, Kubernetes

        User Question: Which candidates know Java?

        Answer: Based on the provided context, the candidates who know Java are Frank and Grace.
        --- Example End ---

        **Answer:**""" # LLM generation starts here
        # --- End Prompt Tuning Iteration 2 ---

        print(f"--- Constructed Tuned Prompt (Iteration 2 - Few Shot) ---")
        # print(prompt) # Uncomment carefully for debugging prompt structure if needed
        print(f"--- End Constructed Tuned Prompt ---")

        # 4. Call the LLM Service
        bot_response = get_ollama_response(prompt=prompt, model="llama3.2:3b")

        print(f"Sending back LLM response: '{bot_response}'")

        # 5. Return the LLM's Response
        return {"reply": bot_response}


# --- Optional: Run with Uvicorn directly ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI server using Uvicorn (directly from script) ---")
    print("--- For development, prefer running: uvicorn app:app --reload --port 8000 --host 0.0.0.0 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)