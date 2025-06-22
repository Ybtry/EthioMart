# EthioMart Named Entity Recognition (NER) System

## Project Overview

This project implements a custom Named Entity Recognition (NER) model using the `spaCy` library in Python. The goal is to extract specific entities – **PRODUCT**, **PRICE**, and **LOCATION** – from Amharic text messages, primarily sourced from Ethiopian Telegram channels related to e-commerce.

This system is designed to help automate the extraction of key information from unstructured text, which can be valuable for market analysis, inventory management, or improving search capabilities in an e-commerce context.

## Key Features

* **Custom NER Model:** Trained on a domain-specific dataset of Amharic text.
* **Entity Extraction:** Identifies PRODUCT, PRICE, and LOCATION entities.
* **Data Preparation Pipeline:** Includes a script to convert raw labeled data into `spaCy`'s optimized `DocBin` format.
* **Model Training & Evaluation:** Utilizes `spaCy`'s training framework to train and evaluate the custom NER model.
## Setup and Installation

To set up and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ybtry/EthioMart.git](https://github.com/Ybtry/EthioMart.git)
    cd EthioMart
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install necessary Python packages:**
    ```bash
    pip install spacy tqdm
    python -m spacy download en_core_web_sm # Download a small English model for blank nlp object
    pip install spacy-lookups-data # Recommended for better tokenization/normalization
    ```
    *(Note: `en_core_web_sm` is implicitly used by `spacy.blank("en")` for basic tokenization rules, even if not part of the training pipeline. `spacy-lookups-data` helps with language-specific normalization tables.)*

## Data Preparation

The `labeled_telegram_product_price_location.txt` file contains your raw labeled data in a CoNLL-like format. The `prepare_ner_data.py` script parses this data and converts it into `spaCy`'s binary `DocBin` format, splitting it into training and development sets.

To prepare the data:
```bash
python3 prepare_ner_data.py