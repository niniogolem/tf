"""Analyzes the frozen bit indices across multiple simulation results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
N = 128        # Longitud del Bloque (Bits totales)
K = 64         # Bits de Información (Tasa = K/N)
n_frozen = N - K  # Número de bits congelados (Los peores canales)

# Nombre de la carpeta (o usa '.' si están en la misma carpeta)
folder_name = 'results_' + str(N) 
file_pattern = '*.csv'

print("--- Configuración del Sistema ---")
print(f"Longitud del Bloque (N): {N}")
print(f"Bits de Información (K): {K}")
print(f"Bits a Congelar (N-K): {n_frozen}")
print("-------------------------------")

# ==========================================
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ==========================================
search_path = os.path.join(folder_name, file_pattern)

# Si la carpeta no existe, buscar en el directorio actual (fallback)
if not os.path.exists(folder_name):
    search_path = file_pattern

file_paths = glob.glob(search_path)

if not file_paths:
    print(f"Error: No se encontraron archivos CSV que coincidan con: {search_path}")
else:
    print(f"Se encontraron {len(file_paths)} archivos.")

data_frames = []
ber_dict = {} 

for i, path in enumerate(file_paths):
    try:
        df = pd.read_csv(path)
        
        # Estandarización de Columnas
        # Asumiendo formato: [Index, BER]
        if df.shape[1] < 2:
            print(f"Saltando {path}: No tiene suficientes columnas.")
            continue
            
        # Forzar nombres de columnas
        df.columns.values[0] = 'index'
        df.columns.values[1] = 'BER'

        # VALIDACIÓN: Verificar si el archivo coincide con N
        if len(df) != N:
            print(f"Advertencia: El archivo {os.path.basename(path)} tiene {len(df)} filas. Se esperaba N={N}. Se omitirá.")
            continue

        # Alineación: Ordenar por índice para asegurar coherencia
        df = df.sort_values('index').reset_index(drop=True)

        # --- LÓGICA DE CONGELAMIENTO (Basada en K y N) ---
        # Seleccionamos los (N-K) índices con la BER MÁS ALTA para congelar
        frozen_indices = df.nlargest(n_frozen, 'BER')['index'].values

        # Marcar como congelado (1) o confiable (0)
        df['is_frozen'] = df['index'].apply(lambda x: 1 if x in frozen_indices else 0)

        # Metadatos para etiquetas
        filename = os.path.basename(path)
        df['source_file'] = filename

        data_frames.append(df)

        # Guardar BER para análisis de correlación (quitando extensión .csv para limpiar el nombre)
        clean_name = os.path.splitext(filename)[0]
        ber_dict[clean_name] = df['BER']

    except Exception as e:
        print(f"Error procesando {path}: {e}")

if not data_frames:
    print("No hay dataframes válidos para procesar.")
    exit()

master_df = pd.concat(data_frames)

# ==========================================
# FIGURA 1: MAPA DE CALOR (ÍNDICES CONGELADOS)
# ==========================================
heatmap_matrix = master_df.pivot(index='source_file', columns='index', values='is_frozen')

# Cálculo de Intersección Común
# Un índice es común si la suma de la columna es igual al número total de archivos
common_mask = (heatmap_matrix.sum(axis=0) == len(data_frames)).astype(int)
heatmap_matrix.loc['COMÚN'] = common_mask

plt.figure(figsize=(15, 8))
# Colores: Claro = Información (Confiable), Azul Marino = Congelado (Ruidoso)
cmap = sns.color_palette("light:navy", as_cmap=True)
sns.heatmap(heatmap_matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor='#eeeeee', yticklabels=True)

plt.title(f'Intersección de Índices Congelados (N={N}, K={K})', fontsize=14)
plt.xlabel('Índice del Canal')
plt.ylabel('Archivo de Origen')
plt.tight_layout()
plt.show()

common_count = common_mask.sum()
print(f"\n[Resultado] Índices Congelados Comunes:{common_count} (El objetivo era {n_frozen})")
if common_count == n_frozen:
    print("ÉXITO: Todos los archivos coinciden exactamente en el conjunto congelado.")
else:
    print(f"ADVERTENCIA: Desacuerdo encontrado. Solo {common_count} índices son universalmente malos.")

# ==========================================
# FIGURA 2: MATRIZ DE CORRELACIÓN Y REGIÓN DE TRANSICIÓN
# ==========================================
ber_df = pd.DataFrame(ber_dict)
# Correlación de Spearman evalúa el ORDEN (Ranking), no los valores absolutos
corr_matrix = ber_df.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0.9, vmax=1.0, fmt=".4f")
plt.title('Correlación de Rango de Spearman (Acuerdo en el Orden del Canal)', fontsize=14)
plt.tight_layout()
plt.show()

# Análisis de la Región de Transición
ranks = ber_df.rank()
rank_variance = ranks.var(axis=1)
top_controversial = rank_variance.nlargest(5)

print("\n[Análisis] Índices más 'Controvertidos' (Mayor Varianza de Rango):")
print("(Estos son los índices donde los canales desacuerdan en su confiabilidad)")
for idx, var in top_controversial.items():
    print(f"Índice {idx}: Varianza {var:.2f}")

# ==========================================
# FIGURA 3: ESPECTRO DE POLARIZACIÓN
# ==========================================
plt.figure(figsize=(12, 6))

for name, ber_series in ber_dict.items():
    sns.kdeplot(
        ber_series, 
        label=name, 
        fill=True, 
        alpha=0.1, 
        linewidth=2,
        bw_adjust=0.3 # Sensibilidad baja para ver los picos
    )

plt.title(f'Espectro de Polarización (Distribución de BER para N={N})', fontsize=14)
plt.xlabel('Tasa de Error de Bit (BER)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.xlim(0, 0.5) # Forzar vista al rango válido de BER
plt.grid(True, linestyle='--', alpha=0.5)
# Mover leyenda afuera si hay muchos archivos
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
