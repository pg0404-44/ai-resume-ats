Features
📄 Resume analysis (PDF/DOCX)
🤖 LLM-based resume & job description parsing
📊 ATS compatibility score
🔍 Resume–Job Description matching
🧠 Skill and keyword gap analysis
💡 Personalized improvement suggestions
📑 PDF report generation
🔐 User authentication
Tech Stack
Frontend: Streamlit
Backend: FastAPI
LLM: GPT-OSS 120B via Groq
NLP: spaCy
Semantic Matching: Sentence Transformers
Authentication: Supabase
Language: Python
How It Works
Resume + Job Description
          ↓
      AI Analysis
          ↓
   ATS Score & Matching
          ↓
 Feedback & Recommendations
Run Locally
1. Install dependencies
pip install -r requirements.txt
2. Add environment variables

Create a .env file:

SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
3. Start Backend
uvicorn backend.main:app --reload --port 8000
4. Start Frontend
streamlit run frontend/streamlit_app.py