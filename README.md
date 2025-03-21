Green Team Chatbot 🌱💬

A Flask-based chatbot that allows users to ask wildlife, biodiversity, and conservation-related questions. The bot retrieves relevant research data from PDFs and generates insights using an LLM (Large Language Model).

📌 Features

✅ WhatsApp-style chat UI (User messages on the right, Bot responses on the left)
✅ FAISS-based semantic search for efficient research retrieval
✅ Minimalist desktop-friendly design with a green-themed bot response
✅ Responsive UI with background image support
✅ HuggingFace LLM Integration for answering queries

🚀 How to Run the Project

1️⃣ Install Dependencies
Ensure you have Python installed, then run:

pip install flask llama-index transformers sentence-transformers faiss-cpu

2️⃣ Clone the Repository

git clone https://github.com/your-repo/chatbot.git
cd chatbot

3️⃣ Add Your HuggingFace API Token
Edit app.py and replace:

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

with your actual Hugging Face API key.

4️⃣ Place Research PDFs
Put your research papers inside the researchPapers folder:

/project-folder
  /researchPapers
    doc1.pdf
    doc2.pdf

5️⃣ Run the Flask App

 app.py

The chatbot will be accessible at:
🔗 http://127.0.0.1:5000/


📜 License
This project is open-source. Feel free to modify and use it for research purposes.

🙌 Contribution
If you have suggestions, feel free to submit a pull request or report issues.

Now you're all set! 🚀 Enjoy using your Green Team Chatbot! 🌿🤖