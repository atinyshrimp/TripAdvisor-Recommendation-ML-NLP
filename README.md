# TripAdvisor Recommendation Challenge - Beating BM25

## Overview
This project was developed as part of the ESILV A5 Machine Learning for NLP course. The objective is to improve recommendation quality for places on TripAdvisor based on review text, surpassing the BM25 model using advanced Natural Language Processing techniques.

## Authors
- **Sarujan Denson**
- **Joyce Lapilus**

## Problem Statement
The challenge is to recommend places on TripAdvisor that are most similar based on review content. The task involves using textual similarity methods to predict ratings and rank places effectively. Our aim is to explore both lexical and semantic models, evaluate their performance, and achieve better accuracy than the BM25 baseline.

## Models Explored
1. **BM25 Model**: A traditional lexical retrieval model using term frequency and inverse document frequency.
2. **TF-IDF Model**: A custom implementation that uses term weighting to calculate similarity.
3. **Embedding-Based Model**: Utilizes SentenceTransformer to obtain dense semantic embeddings for review content.
4. **Hybrid Model**: A combination of TF-IDF for initial candidate retrieval, followed by embedding-based re-ranking for refined results.

## Implementation Details
- **Embedding-Based Model**: Uses `SentenceTransformer` (all-MiniLM-L6-v2) to create high-dimensional vector embeddings and compute cosine similarity.
- **Hybrid Model**: Combines TF-IDF and embedding similarity to balance lexical and semantic matching.

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures the difference between predicted and actual ratings. A lower MSE indicates a more accurate model.
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranks model performance, where a score closer to 1 indicates better ranking alignment with ground truth.

## Results
- **MSE Performance**:
  - Hybrid Model: 22.13% (Best Performance)
  - Embedding-Based Model: 29.76%
  - BM25 Model: 29.91%
  - TF-IDF Model: 31.34%
- **NDCG Performance**:
  - Hybrid Model: 0.9949 (Highest Score)
  - BM25 Model: 0.9933
  - TF-IDF Model: 0.9933
  - Embedding-Based Model: 0.9917

The Hybrid Model outperformed all others, achieving the lowest MSE and the highest NDCG score, demonstrating the effectiveness of combining lexical and semantic similarity approaches.

## File Structure
- `report/ESILV_A5_ML_for_NLP_Project_1_Report.pdf`: The detailed project report.
- `notebooks/recommendation_model.ipynb`: The Jupyter notebook containing code for data processing, model training, and evaluation.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/atinyshrimp/TripAdvisor-Recommendation-Challenge.git
   cd TripAdvisor-Recommendation-Challenge
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/recommendation_model.ipynb
   ```
4. Download NLTK data packages (stopwords, punkt, and wordnet):
   ```bash
   python -m nltk.downloader punkt stopwords wordnet
   ```  
5. Follow the steps in the notebook to train and evaluate the models.
