import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Ubicacion de los archivos CSV
N = 128
folder_name = 'results_' + str(N)
file_pattern = '*.csv'
search_path = os.path.join(folder_name, file_pattern)
file_paths = glob.glob(search_path)

if not file_paths:
    print(f"No se encontraron archivos CSV: {search_path}")
else:
    print(f"Se encontraron {len(file_paths)} archivos.")

data_frames = []

for i, path in enumerate(file_paths):
    df = pd.read_csv(path)
    df.columns.values[0] = 'index'
    df.columns.values[1] = 'BER'

    df = df.sort_values('index').reset_index(drop=True)

    # Determine the number of items to freeze (50% of total rows)    
    # Find the threshold value or get the indices of the largest N values
    n_frozen = len(df) // 2
    frozen_indices = df.nlargest(n_frozen, 'BER')['index'].values

    # Mark as frozen (1) or reliable (0)
    df['is_frozen'] = df['index'].apply(lambda x: 1 if x in frozen_indices else 0)

    # Use the filename (without folder path) as the label
    filename = os.path.basename(path)
    df['source_file'] = filename
    data_frames.append(df)

if data_frames:
    master_df = pd.concat(data_frames)
    
    # Pivot: Rows = Files, Cols = Index, Values = 0 or 1
    heatmap_matrix = master_df.pivot(index='source_file', columns='index', values='is_frozen')
    
    # Calculate Common Intersection (Sum of column must equal number of files)
    common_mask = (heatmap_matrix.sum(axis=0) == len(data_frames)).astype(int)
    heatmap_matrix.loc['COMMON'] = common_mask
    
    plt.figure(figsize=(15, 8))    
    # Colors: White/Light=Reliable, Navy=Frozen
    cmap = sns.color_palette("light:navy", as_cmap=True)
    sns.heatmap(heatmap_matrix, 
                cmap=cmap, 
                cbar=False, 
                linewidths=0.5, 
                linecolor='#eeeeee',
                yticklabels=True)

    plt.title('Frozen Index Intersection (Highest 50% BER)', fontsize=14)
    plt.xlabel('Channel Index')
    plt.ylabel('Source File')
    plt.tight_layout()
    plt.show()
    
    # Optional: Print the exact common indexes for your config
    common_indexes = common_mask[common_mask == 1].index.tolist()
    print(f"Common Frozen Indexes ({len(common_indexes)}): {common_indexes}")