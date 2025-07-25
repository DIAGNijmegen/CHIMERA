import os
import json
import pandas as pd
import torch
import argparse
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ClinicalDataProcessor:
    def __init__(self):
        self.preprocessor = None
        self.output_dim = None
        self.numerical_features = [
            'age_at_prostatectomy',
            'primary_gleason',
            'secondary_gleason',
            'tertiary_gleason',
            'ISUP',
            'pre_operative_PSA'
        ]
        self.categorical_features = [
            'pT_stage',
            'positive_lymph_nodes',
            'capsular_penetration',
            'positive_surgical_margins',
            'invasion_seminal_vesicles',
            'lymphovascular_invasion',
            'earlier_therapy'
        ]

    def fit(self, clinical_df):
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        self.preprocessor.fit(clinical_df)
        self.output_dim = self.preprocessor.transform(clinical_df.head(1)).shape[1]

    def transform(self, clinical_df):
        if self.preprocessor is None:
            raise RuntimeError("Processor has not been fitted yet.")
        return torch.tensor(self.preprocessor.transform(clinical_df), dtype=torch.float32)

def main(args):
    clinical_data = []
    for json_file in Path(args.input_dir).glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['case_id'] = json_file.stem
            clinical_data.append(data)
    
    clinical_df = pd.DataFrame(clinical_data)

    processor = ClinicalDataProcessor()
    processor.fit(clinical_df)

    processed_data = processor.transform(clinical_df)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(processed_data, output_path / 'clinical_features.pt')
    with open(output_path / 'clinical_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    print(f"Clinical data processed and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess clinical data.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw clinical JSON files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed data and processor.')
    args = parser.parse_args()
    main(args)
