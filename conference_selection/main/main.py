import os
import fitz
import logging
from dotenv import load_dotenv
import pandas as pd
import requests
import re
import pathway as pw
from litellm import APIError
from transformers import AutoTokenizer
import json
import csv
import google.generativeai as genai
from collections import defaultdict
from pathway.xpacks.llm.splitters import TokenCountSplitter
import time

load_dotenv()

# Initialize tokenizer for the specified model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load CSV and create a lookup for publishable papers
PUBLISHABLE_LOOKUP = {}
csv_file_path = "sorted_results_df.csv"  # Update with your CSV file path
with open(csv_file_path, mode="r", newline="") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        paper_id = row["paper_id"].strip()
        is_publishable = row["is_publishable_pred"].strip() == "1"
        PUBLISHABLE_LOOKUP[paper_id] = is_publishable

api_key = os.environ.get("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# Mapping of parent folder IDs to conference labels (unchanged)
PARENT_TO_LABEL = {
    "1sJKv0o5ySrigZewU_wtTxysx9j0kO_nV": "KDD",
    "1ZgkbpvhoNKUuH0b4uCv30lyWg3-5ijTC": "NeurIPS",
    "1JVzabziJf4d2drCTXFssFr_wZMnjr8oT": "EMNLP",
    "1RifJJBjm5tA8E20808RjvkIAiWnFbceb": "CVPR",
    "13eDgt0YghQU2qlogGrTrXJzfD0h0F2Iw": "TMLR",
    "1_xFmMlrNDR0wzzPsv6wXXdGz0eX6vaYb": "Non-Publishable",
    "1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u": "Unlabeled",
}


def write_to_csv(paper_id, recommended_conference, rationale):
    file_exists = os.path.exists("unlabeled_results.csv")
    with open("unlabeled_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Paper Id", "Recommended Conference", "Rationale"]
            )  # Add header
        writer.writerow([paper_id, recommended_conference, rationale])


def generate_rationale(
    query_text, recommended_conference, max_attempts=5, delay_seconds=5
):
    prompt = f"""
    RESEARCH PAPER:
    {query_text}

    This research paper has been assigned to the {recommended_conference} conference based on its content and relevance, using a Retrieval-Augmented Generation (RAG) pipeline.

    Please provide a detailed rationale (BETWEEN 130 to 150 WORDS) explaining why this paper is a good fit for the {recommended_conference}. In your rationale, consider the following aspects:
    1.**Methodology** :How the research approach aligns with the themes and focus of the conference.
    2.**Novelty** : Any unique contributions or innovative aspects that make it suitable for the conference.
    3.**Relevance** :How the paper's topic matches the interests and goals of the conference's audience.
    4.**Impact**: The potential influence of the paper in advancing research or practice in the conference's field.

    Ensure that the rationale is **meaningful**, **contextual**, and **concise**, staying within the specified word count range (130-150 WORDS). The explanation should focus on why this paper is an ideal match for the conference's focus and objectives.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Attempt {attempt} - Error generating rationale: {e}")
            if attempt < max_attempts:
                logging.info(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                logging.error("Max attempts reached. Returning failure message.")
                return "Rationale generation failed due to repeated API errors."


def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from a PDF given as bytes.
    """
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
        doc.close()
    except Exception as e:
        logging.error(f"Error processing PDF bytes: {e}")
    return text


def determine_label(parent_folder_id):
    """
    Determine the label based on the parent folder ID using the mapping.
    """
    return PARENT_TO_LABEL.get(parent_folder_id, "Unknown")


def split_text_into_chunks(text, max_tokens=400):
    """Split text into chunks each with at most max_tokens using the tokenizer."""
    class InputSchema(pw.Schema):
        text: str
    df = pd.DataFrame(data={"text": [text, ]})
    text_table = pw.debug.table_from_pandas(df, schema=InputSchema)
    splitter = TokenCountSplitter(max_tokens=max_tokens)
    chunks = text_table.select(chunks=splitter(pw.this.text))
    chunks = pw.debug.table_to_pandas(chunks)
    chunks = chunks['chunks'].to_list()[0]
    chunks_list = [chunk[0] for chunk in chunks]
    return chunks_list


# URL of the API endpoint
api_url = "http://0.0.0.0:8000/v1/retrieve"


def send_chunk_to_api(chunk):
    headers = {"Content-Type": "application/json"}
    params = {"query": chunk, "k": 1}
    while True:
        try:
            response = requests.get(api_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                logging.info(f"API Response: {data}")
                return data
            else:
                raise APIError(f"Failed to fetch data. Status code: {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending chunk to API: {e}")
            logging.error("Trying again in 5 seconds...")
            time.sleep(5)


def on_change(key: pw.Pointer, row: dict, time: int, is_addition: bool):
    """
    Callback function that processes each new/changed PDF file,
    extracts text, splits it into chunks, and sends each chunk to the API.
    """
    if is_addition:
        # Retrieve PDF bytes and raw metadata
        pdf_bytes = row.get("data")
        raw_metadata = row.get("_metadata")

        # Safely parse metadata to a dictionary
        metadata = {}
        try:
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
            else:
                metadata = json.loads(str(raw_metadata))
        except Exception as e:
            logging.error(f"Error parsing metadata: {e}")

        paper_id = metadata.get("name").rstrip(".pdf")

        # Check if the paper is publishable using the CSV lookup
        if paper_id not in PUBLISHABLE_LOOKUP or not PUBLISHABLE_LOOKUP[paper_id]:
            logging.info(
                f"Paper {paper_id} is not publishable. Marking conference and rationale as N/A."
            )
            write_to_csv(paper_id, "N/A", "N/A")  # Log non-publishable papers as N/A
            return  # Skip further processing for non-publishable papers

        parent_folder_id = metadata.get("parent")
        label = determine_label(parent_folder_id)

        text = extract_text_from_pdf_bytes(pdf_bytes)

        # Clean the extracted text
        cleaned_text = re.sub(r"(?<=\.)\n", " \n", text)
        cleaned_text = re.sub(r"(?<!\.\s)\n+", " ", cleaned_text)

        # Split into chunks of at most 400 tokens
        chunks = split_text_into_chunks(cleaned_text, max_tokens=400)

        # Send each chunk to the API
        conference_vote = defaultdict(int)
        for chunk in chunks:
            data = send_chunk_to_api(chunk)
            if data:
                # Safely get conference recommendation from the API response
                parent_id = data[0]["metadata"]["parents"][0]
                recommended_conference = PARENT_TO_LABEL.get(parent_id, "Unknown")
                conference_vote[recommended_conference] += 1

        # Determine final conference based on votes
        if conference_vote:
            final_conference = max(conference_vote, key=conference_vote.get)
        else:
            final_conference = "Unlabeled"

        rationale = generate_rationale(cleaned_text, final_conference)
        write_to_csv(paper_id, final_conference, rationale)
        logging.info(
            f"Processed publishable paper {paper_id}, selected conference {final_conference}."
        )


# Set up reading from Google Drive with metadata
table = pw.io.gdrive.read(
    object_id="1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u",  # main folder ID; adjust if needed
    service_user_credentials_file="credentials.json",
    with_metadata=True,  # Retrieve metadata alongside file data
)

pw.io.subscribe(table, on_change)
pw.run()
