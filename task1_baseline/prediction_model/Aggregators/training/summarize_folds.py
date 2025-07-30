import pandas as pd
from pathlib import Path
import argparse

def summarize_folds(parent_dir: Path):
    """
    Finds all 'summary.csv' files in a specific subdirectory structure,
    concatenates them, adds a 'fold' column, calculates the mean of each
    column, and saves the result to a new CSV file.

    The expected directory structure is:
    parent_dir/
    ├── k=0/
    │   └── .../
    │       └── .../
    │           └── summary.csv
    ├── k=1/
    │   └── .../
    │       └── .../
    │           └── summary.csv
    └── ...

    Args:
        parent_dir (Path): The path to the parent directory containing the 'k=...' folders.
    """
    all_dfs = []
    
    # Find all directories matching the "k=*" pattern
    fold_dirs = sorted(parent_dir.glob("k=*"), key=lambda p: int(str(p.name).split('=')[1]))

    if not fold_dirs:
        print(f"Error: No directories matching 'k=*' found in '{parent_dir}'.")
        return

    print(f"Found {len(fold_dirs)} fold directories.")

    for fold_dir in fold_dirs:
        try:
            # Extract the fold number from the directory name
            fold_number = int(fold_dir.name.split('=')[1])
            
            # Use glob to find the summary.csv file, searching recursively
            # This is more robust if the intermediate folder names change.
            summary_files = list(fold_dir.rglob("summary.csv"))

            if not summary_files:
                print(f"Warning: No 'summary.csv' found in '{fold_dir}'. Skipping.")
                continue
            
            # Assuming there is only one summary.csv per k-folder
            summary_path = summary_files[0]
            
            df = pd.read_csv(summary_path)
            
            # Add the 'fold' column
            df['fold'] = fold_number
            
            all_dfs.append(df)
            print(f"Successfully processed fold {fold_number} from '{summary_path}'.")

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse fold number from directory '{fold_dir.name}'. Skipping. Error: {e}")
        except Exception as e:
            print(f"An error occurred while processing '{fold_dir}': {e}")

    if not all_dfs:
        print("Error: No summary files could be processed. Exiting.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns to have 'fold' first, like the example
    cols = ['fold'] + [col for col in combined_df.columns if col != 'fold']
    combined_df = combined_df[cols]

    # Calculate the mean of all numeric columns
    # The 'fold' column will be ignored automatically by numeric_only=True
    mean_row = combined_df.mean(numeric_only=True).to_frame().T
    mean_row['fold'] = 'mean'  # Label the mean row

    # Append the mean row to the dataframe
    # Use concat instead of append for future compatibility
    final_df = pd.concat([combined_df, mean_row], ignore_index=True)

    # Save the final dataframe to a csv file
    output_path = parent_dir / "all_folds_summary.csv"
    final_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Successfully created summary file at: '{output_path}'")
    print("Final DataFrame preview:")
    print(final_df.head())
    print("...")
    print(final_df.tail())


if __name__ == "__main__":
    # This part allows you to run the script from your terminal
    # Example usage: python your_script_name.py /path/to/your/parent/directory
    
    parser = argparse.ArgumentParser(
        description="Concatenate summary.csv files from k-fold directories and calculate averages."
    )
    parser.add_argument(
        "parent_path", 
        type=str, 
        help="The path to the parent directory containing the 'k=*' folders."
    )
    
    args = parser.parse_args()
    
    parent_directory = Path(args.parent_path)
    
    if not parent_directory.is_dir():
        print(f"Error: The provided path '{parent_directory}' is not a valid directory.")
    else:
        summarize_folds(parent_directory)
