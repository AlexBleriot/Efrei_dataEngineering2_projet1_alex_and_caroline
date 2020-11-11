import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

print("Loadind reviewDataCleaned dataset...")
reviewDataCleaned=pd.read_csv('reviewDataCleaned.csv')

Independent_var=reviewDataCleaned.cleaned_review
Dependant_var=reviewDataCleaned.sentiment
    
X_train, X_test, y_train, y_test = train_test_split(Independent_var, Dependant_var, test_size = 0.20, random_state = 0)

print('Loading model from',MODEL_PATH)
clf= load(MODEL_PATH)

y_predict =clf.predict(X_test)
print(y_predict)