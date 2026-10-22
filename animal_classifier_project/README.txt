ANIMAL CNN CLASSIFIER

1. Put the complete raw-img folder beside train.py and app.py.
   Example:
       animal_classifier_project/
         train.py
         app.py
         requirements.txt
         raw-img/
           cane/
           cavallo/
           elefante/
           farfalla/
           gallina/
           gatto/
           mucca/
           pecora/
           ragno/
           scoiattolo/

2. Use Python 3.10 or 3.11 for TensorFlow 2.16.1 (recommended).
3. Open a terminal in this project folder:
       python -m venv .venv
       .venv\Scripts\activate
       python -m pip install -r requirements.txt
       python train.py
4. When training finishes:
       streamlit run app.py

Training creates:
  models/animal_cnn.keras
  models/class_names.json

For Streamlit Community Cloud:
- Upload app.py, requirements.txt, and the models/ folder to GitHub.
- Do NOT upload raw-img/ unless you specifically want the dataset in the repository.
- Train locally first; the deployed app loads the saved model and does not retrain.
- GitHub has file-size limits, so check the size of animal_cnn.keras before pushing.
