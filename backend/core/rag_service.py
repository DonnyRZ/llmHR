# --- Imports ---
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import torch

# --- Global variables ---
SAMPLE_CANDIDATES_WITH_EMBEDDINGS = []
embedding_model = None

# --- Constants for Retrieval ---
TOP_N = 3
SIMILARITY_THRESHOLD = 0.35 # Keeping the value you found worked better

# --- Initialization Function (Loads model, generates embeddings) ---
def initialize_embeddings():
    global embedding_model, SAMPLE_CANDIDATES_WITH_EMBEDDINGS
    SAMPLE_CANDIDATES_DATA = [
        {"candidate_id": "c1", "candidate_name": "Alice", "skills": ["Python", "SQL", "AWS"], "experience_years": 5, "summary": "Dev focused on backend systems."},
        {"candidate_id": "c2", "candidate_name": "Bob", "skills": ["Java", "Spring", "Docker"], "experience_years": 7, "summary": "Java dev with cloud experience."},
        {"candidate_id": "c3", "candidate_name": "Charlie", "skills": ["Python", "Flask", "React"], "experience_years": 3, "summary": "Full-stack dev, strong in Python."},
        {"candidate_id": "c4", "candidate_name": "Diana", "skills": ["Python", "AWS", "Terraform"], "experience_years": 6, "summary": "Cloud engineer with Python scripting."},
    ]
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    start_time = time.time()
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print(f"Sentence transformer model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load sentence transformer model: {e}")
        embedding_model = None
        SAMPLE_CANDIDATES_WITH_EMBEDDINGS = []
        return

    if embedding_model:
        print("Preparing candidate texts for embedding...")
        candidate_texts_to_embed = []
        for candidate in SAMPLE_CANDIDATES_DATA:
            text = f"Name: {candidate.get('candidate_name', '')}. "
            skills_text = ', '.join(candidate.get('skills', []))
            if skills_text: text += f"Skills: {skills_text}. "
            text += f"Experience: {candidate.get('experience_years', 'N/A')} years. "
            text += f"Summary: {candidate.get('summary', '')}"
            candidate_texts_to_embed.append(text.strip())

        print(f"Generating embeddings for {len(candidate_texts_to_embed)} candidate texts...")
        start_time = time.time()
        try:
            candidate_embeddings_tensor = embedding_model.encode(
                candidate_texts_to_embed,
                show_progress_bar=True,
                convert_to_tensor=True
            )
            embed_time = time.time() - start_time
            print(f"Embeddings generated successfully in {embed_time:.2f} seconds.")

            temp_list = []
            for i, candidate in enumerate(SAMPLE_CANDIDATES_DATA):
                enhanced_candidate = candidate.copy()
                enhanced_candidate['embedding_tensor'] = candidate_embeddings_tensor[i]
                temp_list.append(enhanced_candidate)
            SAMPLE_CANDIDATES_WITH_EMBEDDINGS = temp_list
            print(f"Embeddings stored with candidate data ({len(SAMPLE_CANDIDATES_WITH_EMBEDDINGS)} candidates).")

        except Exception as e:
            print(f"ERROR: Failed during embedding generation: {e}")
            SAMPLE_CANDIDATES_WITH_EMBEDDINGS = []
    else:
        print("Embedding model not available. Skipping embedding generation.")
        SAMPLE_CANDIDATES_WITH_EMBEDDINGS = []

# --- Run initialization on module import ---
initialize_embeddings()


# --- Core Functions ---

# --- MODIFIED FUNCTION DEFINITION AND ADDED PRINT STATEMENT FOR STEP 14.4 ---
def retrieve_context(query: str, analyzed_query: dict = None) -> list[dict]:
    """
    Retrieves the top N most semantically similar candidates based on the user query,
    using cosine similarity of sentence embeddings, above a certain threshold.
    (Will be enhanced in Step 16 to use analyzed_query criteria for filtering).
    """
    global embedding_model, SAMPLE_CANDIDATES_WITH_EMBEDDINGS
    # --- ADDED PRINT STATEMENT ---
    print(f"\n--- retrieve_context called (Step 14 - Param Added). Query: '{query}'. Analyzed: {analyzed_query} ---")

    # Check if embeddings are available
    if not embedding_model or not SAMPLE_CANDIDATES_WITH_EMBEDDINGS:
        print("--- Embeddings or candidate data not available. Returning empty context. ---")
        return []

    try:
        # 1. Generate embedding for the user query
        print("--- Generating query embedding ---")
        query = str(query) if query is not None else ""
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        # 2. Prepare candidate embeddings
        candidate_embeddings = torch.stack([c['embedding_tensor'] for c in SAMPLE_CANDIDATES_WITH_EMBEDDINGS])

        # 3. Compute cosine similarity
        print(f"--- Computing cosine similarity against {len(candidate_embeddings)} candidates ---")
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # 4. Find the top N matches above the threshold
        print(f"--- Finding top {TOP_N} candidates with similarity > {SIMILARITY_THRESHOLD} ---")
        scores_list = list(enumerate(cosine_scores.tolist()))
        scores_list.sort(key=lambda x: x[1], reverse=True)

        top_matches = []
        for index, score in scores_list:
            if score >= SIMILARITY_THRESHOLD and len(top_matches) < TOP_N:
                print(f"--- Match found: Index {index}, Candidate '{SAMPLE_CANDIDATES_WITH_EMBEDDINGS[index]['candidate_name']}', Score {score:.4f} ---")
                match_info = SAMPLE_CANDIDATES_WITH_EMBEDDINGS[index].copy()
                match_info['similarity_score'] = score
                top_matches.append(match_info)
            elif len(top_matches) >= TOP_N:
                 break

        if not top_matches:
            print("--- No candidates met the similarity threshold ---")

        print(f"--- Retrieved {len(top_matches)} relevant candidates ---")
        return top_matches

    except Exception as e:
        print(f"--- Error during semantic retrieval: {e} ---")
        # import traceback
        # traceback.print_exc()
        return []


def format_context_for_llm(candidates: list[dict]) -> str:
    """
    Formats the retrieved candidate data into a string for the LLM prompt.
    Avoids including 'embedding_tensor' and 'similarity_score'.
    Sorts by score if available.
    """
    if not candidates:
        return "No relevant candidate information was found for the query."

    if candidates and 'similarity_score' in candidates[0]:
         try:
             candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
             context_str = "Relevant Candidate Information Found (Ranked by relevance):\n"
         except Exception as e:
             print(f"Warning: Could not sort candidates by score: {e}")
             context_str = "Relevant Candidate Information Found:\n"
    else:
        context_str = "Relevant Candidate Information Found:\n"

    for candidate in candidates:
        context_str += f"- Name: {candidate.get('candidate_name', 'N/A')} \n" # Removed score display
        skills = candidate.get('skills', [])
        if skills:
            context_str += f"  Skills: {', '.join(skills)}\n"
        exp = candidate.get('experience_years', 'N/A')
        context_str += f"  Experience: {exp} years\n"
        summary = candidate.get('summary', 'N/A')
        context_str += f"  Summary: {summary}\n\n"
    return context_str.strip()