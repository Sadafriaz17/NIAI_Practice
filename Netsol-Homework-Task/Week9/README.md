# Week 9 – Research Assistant (dynamic prompt) + Chatbot

## 1. Setup (once)
```bash
cd Week9
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # Windows   (macOS/Linux: cp .env.example .env)
# open .env and set LLM_PROVIDER + the API key (or use ollama = free & local)
```

## 2. Research Assistant (summaries with a dynamic prompt)
```bash
python research_assistant.py sample_papers/sample_paper.txt
python research_assistant.py your_paper.pdf --style beginner --length short --format bullets
python research_assistant.py your_paper.pdf --style technical --length long --focus "methodology and datasets"
python research_assistant.py paper1.pdf paper2.pdf --compare
python research_assistant.py your_paper.pdf --interactive
```
Results are printed and saved in `summaries/`.
The prompt is built at run time from **style** (beginner / technical / executive), **length** (short / medium / long),
**format** (paragraph / bullets / structured) and an optional **focus**. Long papers use map-reduce (notes per part, then a final summary).

## 3. Chatbot (memory + personas + streaming)
```bash
python chatbot.py
python chatbot.py --persona tutor
python chatbot.py --paper your_paper.pdf      # chat about a paper
```
Commands: `/persona tutor|researcher|concise|assistant`, `/reset`, `/history`, `/exit`.
