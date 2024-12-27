import os
import re
import pandas as pd

# Define the folder containing the txt files
folder_path = "Threshold"

# Define the regex pattern to extract data
pattern = {
    "avg_em": re.compile(r"avg_em\s+([\d.]+)"),
    "avg_hits": re.compile(r"avg_hits\s+([\d.]+)"),
    "avg_f1": re.compile(r"avg_f1\s+([\d.]+)"),
    "avg_precision": re.compile(r"avg_precision\s+([\d.]+)"),
    "avg_recall": re.compile(r"avg_recall\s+([\d.]+)")
}

# Initialize a list to store the extracted data
data = []

# Loop through each txt file in the folder
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith(".txt"):
        # Extract the percentage of KG completeness from the filename
        kg_completeness = int(file_name.split(".")[0])
        
        # Read the content of the file
        with open(os.path.join(folder_path, file_name), "r") as file:
            content = file.read()
        
        # Extract the metrics using regex
        row = {"% of KG completeness": kg_completeness}
        for key, regex in pattern.items():
            match = regex.search(content)
            if match:
                row[key] = float(match.group(1))
            else:
                row[key] = None
        
        # Append the row to the data list
        data.append(row)

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Reorder the columns
df = df[[
    "% of KG completeness", "avg_em", "avg_hits", "avg_f1", 
    "avg_precision", "avg_recall"
]]

# Save the DataFrame to a CSV file
output_csv = "kg_completeness_metrics.csv"
df.to_csv(output_csv, index=False)

print(f"Data has been saved to {output_csv}")
