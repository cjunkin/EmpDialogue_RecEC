echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating directories..."
mkdir -p outputs
mkdir -p resources

echo "Downloading GLoVE embeddings..."
test -f resources/glove.6B.zip || wget -P resources https://nlp.stanford.edu/data/glove.6B.zip && unzip -o resources/glove.6B.zip -d resources

echo "Downloading baseline files..."
test -f resources/roberta-large.tsv || wget -P resources https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/en/roberta-large.tsv

echo "Getting trained emotion recognition model..."
mkdir -p outputs/emotion
root_dir=$(realpath ${BASH_SOURCE[0]} | cut -d'/' -f2)
if [[ "$root_dir" == "/content" ]]; then
  echo "On Colab"
  cp /content/CS247-Empathy-Mental-Health/best_emotion.pt outputs/emotion/
else
  echo "On Kaggle"
  cp /kaggle/input/cs247-empathy-mental-health/best_emotion.pt outputs/emotion/
fi

echo "Setup complete."
