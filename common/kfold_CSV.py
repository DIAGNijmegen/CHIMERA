import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_metadata_from_json(train_dir: str) -> pd.DataFrame:
    """Load JSON metadata from each folder under train_dir."""
    records = []
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                json_path = os.path.join(folder_path, file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    slide_id = os.path.splitext(file)[0][:-2]
                    print(slide_id)
                    label = 1 if data.get('BRS') == 'BRS3' else 0
                    data.update({'slide_id': slide_id+'HE', 'label': label})
                    records.append(data)
    return pd.DataFrame(records)

def perform_cross_validation(df: pd.DataFrame, output_dir: str,
                             label_column: str = 'label', n_splits: int = 3) -> None:
    """Perform stratified k-fold and save train/test CSVs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[label_column])):
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        df.iloc[train_idx].to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        df.iloc[test_idx].to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
        print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test samples.")

def main():
    base_dir = r"\\umcn.nl\nas\RBS\PA_CPGARCHIVE\projects\chimera\bladder\task_2\train"
    output_dir = "./splits_chimera"
    df = load_metadata_from_json(base_dir)
    print(f"Total samples loaded: {len(df)}")
    perform_cross_validation(df, output_dir, n_splits = 5)
if __name__ == "__main__":
    main()
