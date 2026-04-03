import pandas as pd
import main_phase1_rl as phase1_rl
import os

agent = phase1_rl.Phase1FlipAgent()

base_folder = "C:/Users/bibhushaojha/Desktop/MMP/instances/p=3"
csv_files = []

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))
            
file_path = csv_files[0]
data = pd.read_csv(file_path)

A, b, c, m, n, d, p = phase1_rl.required_data(data)

result = phase1_rl.main_function(A, b, c, m, n, d, p, phase1_agent=agent)
print(result)
