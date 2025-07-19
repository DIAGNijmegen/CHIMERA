import torch
from training.trainer import load_model  # hypothetical
from wsi_datasets.wsi_classification import WSIClassificationDataset
import pandas as pd

# load test dataset
test_df = pd.read_csv('test.csv')
dataset = WSIClassificationDataset(test_df, ...)

# load trained model
model = load_model('results/fold_0/best_model.pth')
model.eval()

# perform inference
predictions = []
for x in dataset:
    pred = model(x['features'], x['clinical'])
    predictions.append(pred.argmax())

# save to csv
pd.DataFrame({'case_id': test_df['case_id'], 'pred': predictions}).to_csv('predictions.csv', index=False)
