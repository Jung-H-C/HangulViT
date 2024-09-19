import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from model.config import IMG_DIR, LABEL_FILE

def split_dataset(label_file, test_size, shuffle):
    # Train, Test 데이터셋을 불러오는 함수
    labels = pd.read_csv(label_file)

    train_labels, test_labels = train_test_split(labels, test_size=test_size, shuffle=shuffle)

    return train_labels, test_labels

# train_labels, test_labels = split_dataset(LABEL_FILE, test_size = 0.2, shuffle=True)

# 1회성으로만 호출되어야 함!
# with open('test_labels.pkl', 'wb') as f:
#     pickle.dump(test_labels, f)
#
# # 1회성으로만 호출되어야 함!
# with open('train_labels.pkl', 'wb') as f:
#     pickle.dump(train_labels, f)
with open('train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

train_labels, valid_labels = train_test_split(train_labels, test_size = 0.25, shuffle=True)

with open('valid_labels.pkl', 'wb') as f:
    pickle.dump(valid_labels, f)

with open('train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f)

# train, valid, test 를 6:2:2 비율로 설정함.