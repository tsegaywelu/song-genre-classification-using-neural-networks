# Song Genre Classification

This CSE 572 Data Mining project uses neural networks to predict music genres from song lyrics and metadata. The dataset contains **2,500 labeled training songs**, **627 evaluation songs**, and **14 genres**. Each song can have one or more genres.

## Results

| Evaluation | Sample-wise F1 |
| --- | ---: |
| **Official Kaggle score** | **0.75225** |
| Exploratory five-fold cross-validation | 0.7376 |
| Nested cross-validation with model and threshold selection | 0.7341 ± 0.0217 |
| Nested cross-validation with the submitted configuration fixed | 0.7357 ± 0.0196 |

Sample-wise F1 measures how closely the predicted genre set matches the true genre set for each song, then averages the scores across songs. The official Kaggle score comes from the original `submission_candidate.csv`. The other scores come from validation on the labeled training data.

## Approach

- **Single-label classification.** Compared CountVectorizer and TF-IDF features using a neural network with two 128-unit hidden layers and five-fold cross-validation.
- **Multi-label classification.** Retained every genre for each song and compared seven learning rates and three optimizers using sample-wise F1.
- **Final model.** Combined lyric features with artist, release-date, language, and source metadata in a regularized multilayer perceptron. Learned preprocessing and label-based features used only the training portion of each fold.
- **Validation audit.** Used five outer folds and three inner folds to evaluate model and threshold selection on held-out songs.

See [description.pdf](description.pdf) for the full methodology, tables, figures, and Kaggle leaderboard evidence.

## Folder Structure

```text
.
├── README.md
├── code.ipynb
├── description.pdf
└── CSV files/
    ├── train.csv
    ├── evaluation.csv
    └── sample_submission.csv
```

## Run in Google Colab

1. Open [code.ipynb](code.ipynb) in Google Colab and select a Python CPU runtime.
2. Upload `train.csv`, `evaluation.csv`, and `sample_submission.csv` directly to the Colab session’s Files panel.
3. Run all cells from top to bottom. The cross-validation experiments can take several minutes.
4. The final cell creates and downloads `submission_candidate.csv`.

The output contains exactly **627 predictions** with the columns `total` and `predicted_label`. It preserves the song IDs from the submission template and stores each predicted genre set as a JSON list.

