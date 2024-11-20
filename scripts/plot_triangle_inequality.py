import matplotlib.pyplot as plt

# Load violation data from the file
with open('violations.txt', 'r') as f:
    lines = f.readlines()

# The first line contains the proportion of triangles with violations
proportion_line = lines[0].strip()
proportion_value = proportion_line.split(": ")[1]

# The rest of the lines contain the violation values
violations = [float(line.strip()) for line in lines[1:]]

# Plot the cumulative line plot
violations.sort()
plt.plot(violations, range(1, len(violations) + 1), label='Violations')

# Add the proportion to the plot
plt.title(f'Triangle Inequality Violations\n{proportion_value} of triangles had violations')
plt.xlabel('Violation Amount')
plt.ylabel('Number of Violating Triangles')
plt.legend()

plt.savefig('cumulative_violations_plot.png', format='png')
