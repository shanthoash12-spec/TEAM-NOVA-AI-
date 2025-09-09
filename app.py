# ------------------- INSTALL DEPENDENCIES -------------------
!pip install transformers pypdf2 gradio faiss-cpu sentence-transformers

# ------------------- IMPORTS -------------------
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# ------------------- LOAD EMBEDDING MODEL -------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- PDF TEXT EXTRACTION -------------------
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        reader = PyPDF2.PdfReader(file.name)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# ------------------- SPLIT TEXT INTO CHUNKS -------------------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ------------------- LOAD GRANITE MODEL -------------------
model_name = "ibm-granite/granite-3.1-2b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ------------------- VECTOR STORE (FAISS) -------------------
index = None
pdf_chunks = []

def upload_pdfs(files):
    global index, pdf_chunks
    text = extract_text_from_pdfs(files)
    pdf_chunks = chunk_text(text)

    # create embeddings
    embeddings = embedder.encode(pdf_chunks, convert_to_numpy=True)

    # build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return "✅ PDFs uploaded & indexed! Now you can ask questions."

def ask_question(question):
    global index, pdf_chunks
    if index is None:
        return "⚠️ Please upload PDFs first."

    # encode question
    q_emb = embedder.encode([question], convert_to_numpy=True)

    # search top 3 relevant chunks
    D, I = index.search(q_emb, k=3)
    context = "\n".join([pdf_chunks[i] for i in I[0]])

    # pass context to LLM
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].split("Answer:")[-1].strip()

# ------------------- GRADIO APP -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📚 StudyMate – AI-Powered Academic Assistant (with FAISS Retrieval)")
    
    with gr.Tab("Upload PDFs"):
        file_upload = gr.File(file_types=[".pdf"], file_count="multiple")
        upload_button = gr.Button("Process PDFs")
        upload_output = gr.Textbox(label="Status")

        upload_button.click(upload_pdfs, inputs=file_upload, outputs=upload_output)

    with gr.Tab("Ask Questions"):
        question = gr.Textbox(label="Enter your question")
        answer = gr.Textbox(label="Answer")
        ask_button = gr.Button("Get Answer")

        ask_button.click(ask_question, inputs=question, outputs=answer)

# ------------------- LAUNCH APP -------------------
demo.launch()
