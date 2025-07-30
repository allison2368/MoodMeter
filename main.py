import kaggle 
import pandas as pd
# /Users/allisonpeng/.kaggle/kaggle.json

kaggle.api.authenticate()
kaggle.api.dataset_download_files('tapakah68/facial-emotion-recognition', path='data', unzip=True)

# Constants
IMG_SIZE = 48
DATA_DIR = '/kaggle/input/facial-emotion-recognition/images'
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load the dataset
csv_path = '/kaggle/input/facial-emotion-recognition/emotions.csv'
emotions_df = pd.read_csv(csv_path)

