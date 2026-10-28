# Scientific and clinical intent classification with SciBERT

Fine-tunes SciBERT (`allenai/scibert_scivocab_uncased`) on the PubMedQA dataset to judge clinical assertions and hypotheses against biomedical literature. A Streamlit app runs inference in real time.

## Project overview

General-purpose language models often struggle with clinical jargon and drug nomenclature. This project fine-tunes SciBERT, a BERT model pretrained on a 1.14M-paper corpus from Semantic Scholar, to classify biomedical question-context pairs into three classes:

* `YES`: the evidence directly supports the hypothesis or query.
* `NO`: the evidence directly contradicts or refutes the hypothesis.
* `MAYBE`: the clinical evidence is inconclusive, variable, or conditional.

## Architecture and stack

* Base model: `allenai/scibert_scivocab_uncased` (110M parameters, full scientific vocabulary)
* Dataset: PubMedQA (`pqa_labeled` subset, expert-annotated biomedical reasoning pairs)
* Optimization: AdamW with linear learning rate warm-up and FP16 mixed precision
* Frameworks: PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`
* Inference UI: Streamlit
* Hardware: CUDA acceleration on NVIDIA Blackwell and Ada Lovelace GPUs

## Dataset and preprocessing

Each input sample combines the clinical question with its supporting abstract:

$$\text{Input} = \text{"Question: "} + \text{Query} + \text{" Context: "} + \text{Abstract Context}$$

* Tokenizer: sequences are truncated or padded to 256 tokens.
* Split: 80/20 stratified train/test split with random seed 42.
* Metrics: macro-F1 and multi-class accuracy, so that class distribution across the three decisions is accounted for.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Pro1metheus/scientific-intent-scibert.git
cd scientific-intent-scibert
```

### 2. Set up a virtual environment

```bash
# Create and activate the environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Open `model.ipynb` and run all cells. They load PubMedQA, fine-tune SciBERT, and export the weights to `./saved_model`.

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` to try the model.

## Repository structure

```text
scientific-intent-scibert/
│
├── model.ipynb         # Data prep, tokenization, SciBERT fine-tuning, and evaluation
├── app.py              # Streamlit app for live inference
├── requirements.txt    # Frozen Python dependencies
├── .gitignore          # Ignores venv, checkpoints, and model binaries (>100MB)
└── README.md           # Project documentation
```

## License

Released under the [MIT License](LICENSE).
