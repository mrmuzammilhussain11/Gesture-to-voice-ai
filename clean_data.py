import pandas as pd

# 1. Data load karo
file_name = 'data.csv'
df = pd.read_csv(file_name, header=None)
print(f"Original Data Rows: {len(df)}")

# 2. Duplicate rows khatam karo
df.drop_duplicates(inplace=True)

# 3. Check karo ke har row mein 64 columns hain (1 Label + 63 Landmarks)
df = df[df.apply(lambda x: len(x) == 64, axis=1)]

# 4. Data Balancing (Har sign ke maximum 500 records rakho)
# Isse "Gratitude" wala bias khatam ho jayega
df = df.groupby(0).head(500)

# 5. Saaf data save karo
df.to_csv('data_cleaned.csv', index=False, header=False)

print(f"Cleaned Data Rows: {len(df)}")
print("Har sign ke records ab barabar hain. 'data_cleaned.csv' tayyar hai!")