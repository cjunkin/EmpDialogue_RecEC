WORK_DIR=$(cd $(dirname $0); pwd)

echo "Installing dependencies..."
pip install -r "$WORK_DIR/requirements.txt"

echo "Creating directories..."
mkdir -p "$WORK_DIR/outputs"
mkdir -p "$WORK_DIR/resources"

echo "Downloading GLoVE embeddings..."
test -f "$WORK_DIR/resources/glove.6B.zip" || wget -P "$WORK_DIR/resources" https://nlp.stanford.edu/data/glove.6B.zip
unzip -n "$WORK_DIR/resources/glove.6B.zip" -d "$WORK_DIR/resources"

echo "Downloading baseline files..."
test -f "$WORK_DIR/resources/roberta-large.tsv" || wget -P "$WORK_DIR/resources" https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/en/roberta-large.tsv

echo "Getting trained emotion recognition model..."

mkdir -p "$WORK_DIR/outputs/emotion"

root_dir=$(realpath ${BASH_SOURCE[0]} | cut -d'/' -f2)
if [[ "$root_dir" == "content" ]]; then
  echo "On Colab"
  cp /content/CS247-Empathy-Mental-Health/best_emotion.pt "$WORK_DIR/outputs/emotion/"
else
  echo "On Kaggle"
  cp /kaggle/input/cs247-empathy-mental-health/best_emotion.pt "$WORK_DIR/outputs/emotion/"
fi

echo "Setup complete."
