import matplotlib.pyplot as plt
import csv

# Read data from CSV file (relative path)
offsets = []
times = []
with open('../../misaligned_access_results.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 2:
            offsets.append(int(row[0]))
            times.append(float(row[1]))

plt.figure(figsize=(8, 5))
plt.plot(offsets, times, marker='o')
plt.title('Misaligned Access Timing')
plt.xlabel('Offset')
plt.ylabel('Time (ms)')
plt.grid(True)
plt.tight_layout()
plt.show()
