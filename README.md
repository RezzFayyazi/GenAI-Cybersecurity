# Generative AI in Cybersecurity

A hands‑on course for researchers, engineers, and security professionals who want to understand, adapt, and deploy large‑language‑model (LLM) technology in modern cyber‑defense operations.

---

##  📚  What you’ll learn

1. **LLM Foundations** – Transformer architecture, large‑scale pre‑training, and training‑cost trade‑offs.
2. **Model Adaptation** – Supervised fine‑tuning, parameter‑efficient techniques (LoRA, prompt‑tuning) & RLHF with PPO.
3. **Operational Inference** – Prompt‑engineering patterns, Retrieval‑Augmented Generation (RAG) for CVE analysis, evaluation metrics, and guardrails for safe deployment.

Each topic is paired with runnable notebooks or scripts so you can replicate every experiment on your own laptop or in Google Colab.

---

##  🗂️  Repository layout

| Path       | Purpose                                | Run‑time      | Notes                                             |
| ----       | -------------------------------------- | ------------- | ------------------------------------------------- |
| \`Part-1\` | Slides + short Transformer demos       | local         | Training cost & scalability discussion            |
| \`Part-2\` | Fine‑tuning & RLHF                     | Colab / local | Includes fine-tuning and data‑labeling workflow   |
| \`Part-3\` | Prompting, RAG, evaluation, guardrails | Colab / local | End‑to‑end CVE analysis example & evaluation      |


---

## Course Structure

### 01-architecture & pre-training

* `LLM_pretraining.pdf` – Transformer anatomy, training objectives, compute/\$\$ budgeting.
* `Transformers/`

  * `encoder_only.py`
  * `decoder_only.py`
  * `encoder_decoder.py`

### 02‑adaptation

* **Part 1 – Supervised Fine‑Tuning**

  * `LLM_fine-tuning.pdf` - Slides
  * `Finetuning_LLMs_using_LoRA.ipynb` – LoRA, Prompt‑Tuning, adapters.

* **Part 2 – Reinforcement Learning from Human Feedback (RLHF)**

  * `RLHF.pdf` - Slides for RL training for LLMs
  * `RLHF_with_Custom_Datasets.ipynb` - using Label Studio to label your dataset and do RL training

### 03‑inference‑applications

* **Part 1 – Prompt Engineering**

  * `Prompting_Techniques.pdf` - Slides for different prompting techniques
  * `Prompt Templates.docx` – Some cybersecurity-oriented prompt templates
  * `LLM_tutorial.ipynb` – A colab notebook on how to use GPT and Gemini models using the Prompt Templates

* **Part 2 – Retrieval Augmented Generation**

  * `RAG.pdf` - Slides for differnet retrieval techniques
  * `RAG_tutorial_with_CVEs.ipynb` - A colab notebook on how to use RAG techniques for out-of-distribution data

* **Part 3 – LLM Evaluation**

  * `LLM_Evaluation.pdf` - Slides for the evaluation metrics
  * `analysis_main.py` – Code to use Rouge, BLEU, Embedding Similarity metrics for evaluating the responses

* **Part 4 – Guardrails**

  * `Guardrails.pdf` - Slides for different types of guardrails applied to LLMs.
  * `guardrails.ipynb` - Code on how to apply guardrails


---

##  © License

* Code: MIT License (see `LICENSE`)
* Course Materials (slides, docs): CC BY 4.0 (see `LICENCE-content`) 
