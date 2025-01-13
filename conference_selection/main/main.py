import os
import fitz
import io
import logging
import pytesseract
from PIL import Image
import requests
import re
import pathway as pw
from transformers import AutoTokenizer
import json
import csv
import google.generativeai as genai
from collections import defaultdict
from pathway.xpacks.llm.splitters import TokenCountSplitter

# Initialize tokenizer for the specified model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Toggle OCR if necessary
USE_OCR = False

api_key = "AIzaSyDQ_s3CR_zqfbAzBfbLOr8ziXu6MzPf_f0"
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


def generate_rationale(query_text, recommended_conference):
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
    response = response = model.generate_content(prompt)
    return response.text


def extract_text_from_pdf_bytes(pdf_bytes, use_ocr=False):
    """
    Extract text from a PDF given as bytes.
    If use_ocr is True and no text is extracted, perform OCR on each page.
    """
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            elif use_ocr:
                # Render page to image for OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                text += ocr_text
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
    tokens = tokenizer.encode(text)
    chunks = []
    # splitter = pw.xpacks.llm.splitters.TokenCountSplitter(max_tokens=max_tokens)
    # chunks = splitter(text)
    # logging.info(type(chunks))  # debug
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


# URL of the API endpoint
api_url = "http://0.0.0.0:8000/v1/retrieve"


def send_chunk_to_api(chunk):
    headers = {"Content-Type": "application/json"}
    params = {"query": chunk, "k": 1}
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        # Process the returned data as needed
        logging.info(f"type(data): {type(data)}")
        logging.info(f"API Response: {data}")
        return data
    else:
        logging.error(f"Failed to fetch data. Status code: {response.status_code}")
        return ""


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

        parent_folder_id = metadata.get("parent")
        # Label determination remains for potential future use
        label = determine_label(parent_folder_id)

        text = extract_text_from_pdf_bytes(pdf_bytes, use_ocr=USE_OCR)

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
                recommended_conference = PARENT_TO_LABEL[
                    data[0]["metadata"]["parents"][0]
                ]
                conference_vote[recommended_conference] += 1
        final_conference = max(conference_vote, key=conference_vote.get)
        # recommended_conference = PARENT_TO_LABEL[data[0]["metadata"]["parents"][0]]
        rationale = generate_rationale(cleaned_text, final_conference)
        write_to_csv(data[0]["metadata"]["name"], final_conference, rationale)
        logging.info(f"Processed file {metadata.get('name')}, sent all chunks to API.")


# Set up reading from Google Drive with metadata
table = pw.io.gdrive.read(
    object_id="1Pct8i0Tvok1hcsuuRhx_BJQkuFJYaTjq",  # main folder ID; adjust if needed
    service_user_credentials_file="credentials.json",
    with_metadata=True,  # Retrieve metadata alongside file data
)

pw.io.subscribe(table, on_change)
pw.run()
