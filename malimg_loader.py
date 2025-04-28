import os
import csv

# Mapping from Family name to ClassID
family_to_classid = {
    "Allaple.L": 0,
    "Allaple.A": 1,
    "Yuner.A": 2,
    "Lolyda.AA1": 3,
    "Lolyda.AA2": 4,
    "Lolyda.AA3": 5,
    "C2LOP.P": 6,
    "C2LOP.gen!g": 7,
    "Instantaccess": 8,
    "Swizzor.gen!I": 9,
    "Swizzor.gen!E": 10,
    "VB.AT": 11,
    "Fakerean": 12,
    "Alueron.gen!J": 13,
    "Malex.gen!J": 14,
    "Lolyda.AT": 15,
    "Adialer.C": 16,
    "Wintrim.BX": 17,
    "Dialplatform.B": 18,
    "Dontovo.A": 19,
    "Obfuscator.AD": 20,
    "Agent.FYI": 21,
    "Autorun.K": 22,
    "Rbot!gen": 23,
    "Skintrim.N": 24,
}

# Path to your dataset root directory
dataset_dir = "malimg_dataset/malimg_families"

# Output CSV file path
output_csv = "malimg_labels.csv"

rows = []

# Traverse through each family directory
for family, class_id in family_to_classid.items():
    family_dir = os.path.join(dataset_dir, family)
    if not os.path.isdir(family_dir):
        print(f"Warning: Directory not found for family {family}")
        continue
    for filename in os.listdir(family_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # image extensions
            file_id = os.path.splitext(filename)[0]
            rows.append([file_id, class_id])

# Write to CSV
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'class'])
    writer.writerows(rows)

print(f"CSV file '{output_csv}' created with {len(rows)} entries.")
