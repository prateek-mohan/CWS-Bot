from flask import Flask, render_template, request, jsonify
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize a cache for prompts and responses
prompt_cache = {}

# Load LLM
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return HuggingFaceLLM(model=model, tokenizer=tokenizer)

llm = load_llm()

# Extract and split text from PDFs
def extract_and_split_text_from_pdfs(folder_path, chunk_size=1024, chunk_overlap=20):
    extracted_texts = []
    
    def process_pdf(pdf_path):
        try:
            reader = SimpleDirectoryReader(input_files=[pdf_path])
            documents = reader.load_data()
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = splitter.get_nodes_from_documents(documents)
            return nodes
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return []

    with ThreadPoolExecutor() as executor:
        pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
        futures = {file: executor.submit(process_pdf, file) for file in pdf_files}

        for file, future in futures.items():
            nodes = future.result()
            for node in nodes:
                extracted_texts.append(node.text)

    return extracted_texts

# Create FAISS index
def create_faiss_index_from_texts(texts):
    embedder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, embedder

# Query FAISS index
def query_faiss(index, embedder, query_text, texts, k=3):
    query_embedding = embedder.encode([query_text], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    retrieved_paragraphs = [texts[i] for i in indices[0]]
    retrieved_info = "\n".join(retrieved_paragraphs)
    return retrieved_paragraphs, retrieved_info

# Generate LLM response
def generate_research_insights(llm, retrieved_info, query_text):
    cache_key = (query_text, retrieved_info)
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]

    prompt = (f"Context: {retrieved_info} \nQuestion: {query_text}\n"
              f"You are a wildlife expert, known for your ability to give concise, concrete and to the point answers relevant to wildlife"
              f"If the context doesn't contain relevant information, say 'I don't have enough information to answer that question'.")

    response = llm.complete(prompt, max_tokens=200, temperature=0.6)
    insights = response.text
    prompt_cache[cache_key] = insights
    return insights

# Load PDFs and index at startup
folder_path = "/Users/pratemoh/Desktop/green team/grrenhack/researchPapers"
extracted_texts = extract_and_split_text_from_pdfs(folder_path)
index, embeddings, embedder = create_faiss_index_from_texts(extracted_texts)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query_text = data.get("query", "")

    retrieved_paragraphs, retrieved_info = query_faiss(index, embedder, query_text, extracted_texts)
    insights = generate_research_insights(llm, retrieved_info, query_text)

    return jsonify({"response": insights})

if __name__ == "__main__":
    app.run(debug=True)
