pip install -r ../requirements.txt

wget https://nlp.stanford.edu/data/glove.6B.zip
unzip -o glove.6B.zip

mkdir EmpDialogue_RecEC/outputs
mkdir EmpDialogue_RecEC/outputs/emotion

wget https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/en/roberta-large.tsv baseline.tsv

if [ ! -d "/kaggle/working" ]; then
  echo "On kaggle"
  cp /kaggle/input/cs247-empathy-mental-health/best_emotion.pt EmpDialogue_RecEC/outputs/emotion/
else
  echo "On colab"
  cp /content/CS247-Empathy-Mental-Health/best_emotion.pt EmpDialogue_RecEC/outputs/emotion/
fi

echo "Setup done."
