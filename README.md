# MCDM Weight-Slider App

An interactive page that ranks the four classifiers (Decision Tree, Bagging, AdaBoost,
Random Forest) with four MCDM methods (TOPSIS, MARCOS, COCOSO, MABAC). Move the
importance sliders and watch the ranking respond. It is built so a reader who has never
heard of MCDM can follow it.

It needs only the metrics file from your Colab run. It does not need the dataset or the
trained models.


## Run it on your computer
```
pip install -r requirements.txt
streamlit run app.py
```

## Use your own numbers
By default the app shows the reference run. After your Colab run, take
`model_evaluation_metrics.json` and upload it in the sidebar. If the file is malformed the
app says so and falls back to the reference run.

## Put it online (free)
1. Push `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to share.streamlit.io, choose "New app", pick the repo, and set the main file to `app.py`.
   (Hugging Face Spaces with the Streamlit SDK works the same way.)

## What it shows
- Five weight presets, including the stakeholder weights from the paper, plus 8 manual sliders.
- Rank of each model under each method, with an average rank and a winner banner.
- The raw metric table (best value per criterion highlighted).
- A robustness check: 100 random ±30% changes to the weights, showing how often each model wins.

## Honest caveats (also shown in the app)
- MCDM ranks models only on the criteria and weights you give it. A different set of weights gives a different answer.
- All four models score about 70% accuracy here, so the ranking depends mostly on training time, complexity and interpretability.
- Interpretability is coded 1 (fully) or 2 (partially) so that MARCOS avoids dividing by zero.
