echo "Installing dependencies..."
pip install -r EmpDialogue_RecEC/requirements.txt

echo "Get GLoVE embeddings..."
wget https://nlp.stanford.edu/data/glove.6B.zip
mv glove.6B.zip glove.6B.zip
unzip -o glove.6B.zip

echo "Create directories..."
mkdir EmpDialogue_RecEC/outputs
mkdir EmpDialogue_RecEC/outputs/emotion

echo "Get baseline files..."
wget https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/en/roberta-large.tsv baseline.tsv

echo "Move files..."
current_dir=$(basename "$PWD")

if [ "$current_dir" == "content" ]; then
  echo "On colab"
  cp /content/CS247-Empathy-Mental-Health/best_emotion.pt EmpDialogue_RecEC/outputs/emotion/
else
  echo "On kaggle"
  cp /kaggle/input/cs247-empathy-mental-health/best_emotion.pt EmpDialogue_RecEC/outputs/emotion/
fi

echo "Setup done."
