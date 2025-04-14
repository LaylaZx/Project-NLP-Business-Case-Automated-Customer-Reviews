# Project-NLP-Business-Case-Automated-Customer-Reviews
This project consists of three main parts:

1. **Sentiment Analysis** (BERT-based classification of reviews into Negative/Neutral/Positive)
2. **Product Clustering** (AI-powered grouping of similar products using semantic embeddings)
3. **Review Summarization** (Generative AI insights generation)

## Part 1: Sentiment Analysis with Bert
This part implements a BERT-based sentiment classifier to categorize product reviews into negative (0), neutral (1), or positive (2) sentiments. The model is fine-tuned on a balanced dataset and evaluated using standard NLP metrics.
✔ Fine-tuned BERT (bert-base-uncased) for 3-class sentiment classification
✔ Handles class imbalance via oversampling
✔ Tracks training/validation loss
✔ Evaluates performance with:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix

## Part 2: Product Clustering
This part groups similar products to uncover patterns in customer feedback and streamline product analytics.
✔ Uses Sentence-BERT (all-MiniLM-L6-v2) for semantic embeddings
✔ Groups similar products via K-Means clustering
✔ Visualizes clusters in 2D using PCA
✔ Identifies top-performing products per category

## Part 3: AI Review Summarization
This part summarizes customer feedback and generates business-intelligent reports using generative models.
✔ Leverages ChatGPT-3.5 for automated summarization and insights generation
✔ Includes performance benchmarking of summarized content
✔ Detects recurring complaint patterns to inform business strategy
✔ Generates business-ready reports suitable for stakeholders and decision-maker
