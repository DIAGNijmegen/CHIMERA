import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ClinicalDataProcessor:
    """
    Processor for clinical data. Handles loading, preprocessing, and feature extraction
    from clinical JSON files.
    """
    def __init__(self, clinical_data_path=None):
        """
        Initialize the clinical data processor
        
        Args:
            clinical_data_path: Path to directory containing clinical JSON files
        """
        self.clinical_data_path = clinical_data_path
        self.preprocessor = None
        
        # Define features to use from clinical data
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
        
        # Default values for missing data
        self.default_values = {
            'age_at_prostatectomy': 0,
            'primary_gleason': 0,
            'secondary_gleason': 0,
            'tertiary_gleason': 0,
            'ISUP': 1,
            'pre_operative_PSA': 0.0,
            'pT_stage': 'x',
            'positive_lymph_nodes': 'x',
            'capsular_penetration': 'x',
            'positive_surgical_margins': 'x',
            'invasion_seminal_vesicles': 'x',
            'lymphovascular_invasion': 'x',
            'earlier_therapy': 'none'
        }
        
        # Output dimension (will be set after fitting)
        self.output_dim = None
    
    def fit(self, case_ids):
        """
        Fit the preprocessor on clinical data for the given case IDs
        
        Args:
            case_ids: List of case IDs to fit on
        
        Returns:
            self
        """
        if self.clinical_data_path is None:
            print("Warning: No clinical data path specified. Processor will return default values.")
            self.output_dim = self._get_default_output_dim()
            return self
        
        # Load clinical data for all case IDs
        clinical_data = []
        for case_id in case_ids:
            data = self._load_clinical_data(case_id)
            clinical_data.append(data)
        
        # Create a DataFrame from the clinical data
        clinical_df = pd.DataFrame(clinical_data)
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified in the transformers
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(clinical_df)
        
        # Store the output dimension
        self.output_dim = self._get_output_dim(clinical_df)
        print(f"Clinical preprocessor fitted. Output dimension: {self.output_dim}")
        
        return self
    
    def transform(self, case_id):
        """
        Transform clinical data for a single case ID
        
        Args:
            case_id: Case ID to transform
        
        Returns:
            Tensor of processed clinical features
        """
        if self.preprocessor is None or self.clinical_data_path is None:
            # Return zero vector with correct dimension if no preprocessor or clinical data path
            zeros = torch.zeros(self._get_default_output_dim())
            return zeros
        
        # Load clinical data
        data = self._load_clinical_data(case_id)
        
        # Create a DataFrame with a single row
        df = pd.DataFrame([data])
        
        # Transform the data
        try:
            processed_features = self.preprocessor.transform(df)
            # Convert to tensor
            return torch.tensor(processed_features, dtype=torch.float32).squeeze(0)
        except Exception as e:
            print(f"Error transforming clinical data for case {case_id}: {e}")
            # Return zeros as fallback
            return torch.zeros(self.output_dim)
    
    def _get_default_output_dim(self):
        """
        Get the default output dimension for when no preprocessor is available
        
        Returns:
            Default output dimension
        """
        if hasattr(self, 'output_dim') and self.output_dim is not None:
            return self.output_dim
        else:
            # Estimate output dimension based on feature counts
            num_features = len(self.numerical_features)
            
            # Estimate categorical dimensions
            cat_features = 0
            for feature in self.categorical_features:
                if feature == 'pT_stage':
                    cat_features += 9  # Assuming all pT_stage values
                elif feature == 'earlier_therapy':
                    cat_features += 5  # Assuming all therapy types
                else:
                    cat_features += 3  # Assuming binary + unknown
            
            return num_features + cat_features
    
    def _get_output_dim(self, example_df):
        """
        Get the output dimension by transforming an example dataframe
        
        Args:
            example_df: Example DataFrame to transform
        
        Returns:
            Output dimension
        """
        # Transform the example dataframe and get the shape
        transformed = self.preprocessor.transform(example_df.iloc[:1])
        return transformed.shape[1]
    
    def _load_clinical_data(self, case_id):
        """
        Load clinical data for a single case ID
        
        Args:
            case_id: Case ID to load
        
        Returns:
            Dictionary of clinical data
        """
        if self.clinical_data_path is None:
            return self.default_values
        
        json_path = os.path.join(self.clinical_data_path, f"{case_id}.json")
        
        if not os.path.exists(json_path):
            print(f"Clinical data file not found for case {case_id}: {json_path}")
            return self.default_values
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Filter to include only the features we want and provide defaults for missing values
            filtered_data = {}
            for feature in self.numerical_features + self.categorical_features:
                value = data.get(feature, self.default_values[feature])
                # --- Custom preprocessing for each feature ---
                if feature in ['primary_gleason', 'secondary_gleason', 'tertiary_gleason', 'ISUP']:
                    # Ordinal integer, treat 'x' or missing as 0 (or np.nan if you prefer)
                    try:
                        value = int(value)
                    except Exception:
                        value = 0
                elif feature == 'pT_stage':
                    # Map to ordinal integer (example mapping, adjust as needed)
                    pt_stage_map = {
                        '2': 0, '2a': 1, '2b': 2, '2c': 3,
                        '3': 4, '3a': 5, '3b': 6,
                        '4': 7, '4a': 8, '4b': 9, 'x': -1
                    }
                    value = pt_stage_map.get(str(value).lower(), -1)
                elif feature in ['positive_lymph_nodes', 'capsular_penetration', 'invasion_seminal_vesicles']:
                    # Binary with 'x' for unknown
                    if str(value) == '1':
                        value = 1
                    elif str(value) == '0':
                        value = 0
                    else:
                        value = -1
                elif feature == 'positive_surgical_margins':
                    # Binary with 'x' for unknown
                    try:
                        value = int(value)
                        if value not in [0, 1]:
                            value = -1
                    except Exception:
                        value = -1
                elif feature == 'lymphovascular_invasion':
                    # Binary with 'x' for unknown, sometimes float string
                    if str(value) in ['1', '1.0']:
                        value = 1
                    elif str(value) in ['0', '0.0']:
                        value = 0
                    else:
                        value = -1
                # earlier_therapy: keep as string for one-hot
                filtered_data[feature] = value
            
            return filtered_data
        except Exception as e:
            print(f"Error loading clinical data for case {case_id}: {e}")
            return self.default_values
