
import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the JSON file
file_path = F'../data/strategy/{input(">")}/data.json'
data = pd.read_json(file_path)

# Define the years range
years = range(1998, 2009)


plt.figure(figsize=(15, 8))


for start_year in years:
    try:
        returns = [data[f'SplitBy{year}']['YearReturnList'][0] if year >= start_year else None for year in years]
        plt.plot(years, returns, marker='o', label=f'Start Year: {start_year}')
    except KeyError:
        pass

plt.xlabel('Year')
plt.ylabel('Return')
plt.title('Yearly Returns for Different Data Sets Starting from Various Years')
plt.legend()
plt.grid(True)
plt.show()