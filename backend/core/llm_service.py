import requests
import os
import json # Make sure json is imported
from dotenv import load_dotenv

# Load environment variables (optional, good practice)
load_dotenv()
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_CHAT_API_URL = f"{OLLAMA_ENDPOINT}/api/chat" # Using /api/chat which supports messages format

# ======================================================
# == FUNCTION TO GET OLLAMA RESPONSE (Corrected) ==
# ======================================================
def get_ollama_response(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    Sends a prompt to the Ollama API /api/chat endpoint and returns the LLM's response string.
    Handles potential connection errors and extracts the message content.
    Returns an error message string if something goes wrong.
    Dynamically adjusts payload based on prompt type (analysis vs RAG).
    """
    print(f"--- Sending request to Ollama API (Model: {model}) ---")
    # print(f"--- Prompt Start ---\n{prompt}\n--- Prompt End ---") # Uncomment for verbose debugging

    try:
        # Base payload structure
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        # --- ** FIX: Adjust payload based on prompt type ** ---
        # Check if it's the analysis prompt
        is_analysis_prompt = ("Possible Intents:" in prompt and "JSON Response:" in prompt)

        if is_analysis_prompt:
             payload["format"] = "json" # Explicitly request JSON for analysis
             print("--- Requesting JSON format from Ollama for analysis prompt ---")
        else:
             # For regular RAG prompts, ensure 'format' is not present or default
             payload.pop("format", None) # Remove format key if it exists
             print("--- Requesting default format from Ollama for RAG prompt ---")
        # --- ** END FIX ** ---


        headers = {'Content-Type': 'application/json'}
        print(f"--- Sending payload: {json.dumps(payload, indent=2)} ---") # Log the payload being sent
        response = requests.post(OLLAMA_CHAT_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Handle response based on whether JSON format was requested
        if is_analysis_prompt:
             # When format=json, the response content might be directly in message.content as a stringified JSON
             try:
                 message_content_str = data.get('message', {}).get('content')
                 if message_content_str:
                     # Return the raw string - parsing happens in analyze_query_intent
                     print("--- Received JSON formatted response string from Ollama ---")
                     return message_content_str.strip()
                 else:
                     # Log the actual response if content is missing
                     print(f"--- Ollama response missing content field (JSON format requested). Full response: {data} ---")
                     raise ValueError("Content field missing in Ollama JSON response")
             except Exception as e:
                  print(f"--- Error processing JSON formatted response: {e}. Full response: {data} ---")
                  return f"Error: Could not process JSON response from LLM. Details: {e}"
        else:
            # Standard extraction for non-JSON format (natural language) responses
            message_content = data.get('message', {}).get('content')
            if message_content:
                print("--- Received valid text response from Ollama ---")
                return message_content.strip()
            else:
                print(f"--- Ollama response missing content field. Full response: {data} ---")
                return "Error: Received an empty response from the language model."


    except requests.exceptions.ConnectionError as e:
        print(f"--- Ollama Connection Error: {e} ---")
        return f"Error: Could not connect to Ollama service at {OLLAMA_CHAT_API_URL}. Is Ollama running?"
    except requests.exceptions.RequestException as e:
        print(f"--- Ollama Request Error: {e} ---")
        error_detail = str(e)
        if e.response is not None:
            try:
                # Log the error response text from Ollama if available
                print(f"--- Ollama Error Response Body: {e.response.text} ---")
                error_detail += f" | Response: {e.response.text}"
            except Exception:
                pass # Ignore if response text isn't readable
        return f"Error: Failed to get response from Ollama. {error_detail}"
    except json.JSONDecodeError as e:
        print(f"--- Ollama JSON Decode Error: {e} ---")
        print(f"--- Raw response text from Ollama: {response.text if 'response' in locals() else 'N/A'} ---")
        return "Error: Could not understand the response format from Ollama."
    except Exception as e:
        print(f"--- An unexpected error occurred in get_ollama_response: {e} ---")
        # import traceback
        # traceback.print_exc()
        return "Error: An unexpected error occurred while processing the LLM request."


# ======================================================
# == QUERY ANALYSIS FUNCTION (FROM STEP 14.2) ==
# ======================================================
def analyze_query_intent(user_query: str) -> dict:
    """
    Uses the LLM to analyze the user's query, identify intent, and extract criteria.
    Returns a dictionary with the structured analysis.
    """
    print(f"\n--- Analyzing Query Intent for: '{user_query}' ---")

    analysis_prompt = f"""Analyze the following user query to understand the primary intent and extract relevant criteria mentioned for searching candidate data.

User Query: "{user_query}"

Possible Intents (Choose one that best fits):
- find_candidates: User wants a list/information about candidates matching criteria.
- compare_candidates: User wants to compare two or more specific candidates.
- summarize_candidate: User wants details/summary about one specific named candidate.
- general_query: User is asking a general question not specific to filtering/comparing candidates based on the provided criteria types (handle carefully, may need clarification).
- unknown: The intent is unclear or doesn't fit other categories.

Criteria to Extract (Extract only if explicitly mentioned or clearly implied):
- skills: List of required technical or soft skills mentioned (e.g., ["Python", "AWS", "leadership"]). If none, use empty list [].
- experience_years_min: Minimum years of experience required (as an integer). If none mentioned or unclear, use null.
- candidate_names: List of specific candidate names mentioned (e.g., ["Alice", "Bob"]). If none, use empty list [].
- (Add other criteria like education, job titles later if needed)

Instructions:
1. Carefully read the 'User Query'.
2. Determine the most likely 'intent' from the 'Possible Intents' list.
3. Extract any mentioned 'criteria' according to the definitions above. Be precise.
4. Format your response *only* as a single, valid JSON object.
5. The JSON object MUST contain the keys 'intent' (string) and 'criteria' (object).
6. The 'criteria' object should contain keys for the criteria found (e.g., 'skills', 'experience_years_min', 'candidate_names').
7. Do NOT include any explanations, apologies, or conversational text before or after the JSON object.

Example JSON Output format:
{{
  "intent": "find_candidates",
  "criteria": {{
    "skills": ["Python", "React"],
    "experience_years_min": 3,
    "candidate_names": []
  }}
}}

Respond ONLY with the JSON object:
"""

    # Use the modified get_ollama_response function, which now requests JSON format for this prompt
    raw_analysis_response = get_ollama_response(prompt=analysis_prompt, model="llama3.2:3b") # Explicitly passing model again

    print(f"--- Raw Analysis Response from LLM: {raw_analysis_response} ---")

    # --- Attempt to parse the LLM's response string as JSON ---
    structured_analysis = {
        "intent": "unknown",
        "criteria": {},
        "original_query": user_query
    }

    if raw_analysis_response and isinstance(raw_analysis_response, str) and not raw_analysis_response.startswith("Error:"):
        try:
            cleaned_response = raw_analysis_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:-3].strip()

            parsed_json = json.loads(cleaned_response)

            if isinstance(parsed_json, dict):
                structured_analysis["intent"] = parsed_json.get("intent", "unknown")
                criteria = parsed_json.get("criteria", {})
                if isinstance(criteria, dict):
                    structured_analysis["criteria"]["skills"] = criteria.get("skills", [])
                    exp_years = criteria.get("experience_years_min")
                    try:
                        structured_analysis["criteria"]["experience_years_min"] = int(exp_years) if exp_years is not None and str(exp_years).isdigit() else None
                    except (ValueError, TypeError):
                         structured_analysis["criteria"]["experience_years_min"] = None

                    structured_analysis["criteria"]["candidate_names"] = criteria.get("candidate_names", [])
                else:
                     print("--- Warning: 'criteria' field in LLM JSON response was not an object. ---")
            else:
                 print("--- Warning: LLM JSON response was not an object. ---")

            print(f"--- Parsed Query Analysis: {structured_analysis} ---")

        except json.JSONDecodeError:
            print(f"--- Error: LLM analysis response was not valid JSON. Raw: '{raw_analysis_response}'. Using default analysis. ---")
        except Exception as e:
            print(f"--- Error processing LLM analysis response: {e}. Raw: '{raw_analysis_response}'. Using default analysis. ---")
    else:
         print(f"--- LLM call for analysis failed or returned unexpected type: {type(raw_analysis_response)}. Using default analysis. ---")

    return structured_analysis