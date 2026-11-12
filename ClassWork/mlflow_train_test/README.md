# MLflow Train Test Project

Steps:

1. Create a virtual environment (optional)
   python -m venv venv
   venv\Scripts\activate

2. Install packages
   pip install -r requirements.txt

3. Train and log
   python train.py

4. Open the MLflow dashboard
   mlflow ui
   Then open http://127.0.0.1:5000 in your browser.

Open the experiment named train_test_demo. You will see train and test metrics,
parameters, a confusion matrix picture, and the saved model.

To try new settings, edit the values at the top of train.py and run it again.
Each run is saved so you can compare them.
