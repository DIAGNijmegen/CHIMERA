import os
from pathlib import Path

def backup_all_files(root_dir):
    """
    Backs up all non-image, text-based files in a directory and its
    subdirectories into a single text file.
    """
    folder_name = Path(root_dir).name
    output_filename = f'{folder_name}.txt'
    output_txt_path = Path(root_dir) / output_filename
    
    # 🔽 Define a tuple of image file extensions to ignore
    IMAGE_EXTENSIONS = (
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp'
    )

    with open(output_txt_path, 'w', encoding='utf-8') as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                # 🔽 Updated filter to also skip image files
                # Using .lower() makes the check case-insensitive (e.g., ignores .JPG and .jpg)
                if (filename.endswith('.pyc') or 
                    filename == output_filename or 
                    filename.lower().endswith(IMAGE_EXTENSIONS)):
                    continue

                file_path = os.path.join(dirpath, filename)
                out.write(f'# {file_path}\n')
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f'[Could not read file: {e}]\n')
                out.write('\n\n')

if __name__ == '__main__':
    # ⚠️ Remember to change this to your target directory
    pathology_dir = '/Users/robertspaans/Documents/Projects/phd_projects/multimodal_working_group/MICCAI2025/MICCAI2025_models/CHIMERA/task1_baseline'
    backup_all_files(pathology_dir)
    output_file = Path(pathology_dir) / (Path(pathology_dir).name + '.txt')
    print(f"✅ Backup complete. All non-image files from '{pathology_dir}' have been saved to '{output_file}'")