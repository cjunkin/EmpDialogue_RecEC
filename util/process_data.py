import csv

# Open the input text file
with open('../reddit_result.txt', 'r') as input_file:
    lines = input_file.readlines()

# Initialize lists to store S, T, and H lines
s_lines = []
t_lines = []
h_lines = []

# Iterate through each line and append to the appropriate list
for line in lines:
    if "INFO]" in line:  # Check if the line contains "INFO]"
        _, line_content = line.split("INFO]")  # Split the line at "INFO]"
        line_content = line_content.strip()
        if line_content.startswith('S-') or line_content.startswith('T-') or line_content.startswith('H-'):
            # Find the index of the first tab character, which separates the identifier from the content
            tab_index = line_content.find('\t')
            if tab_index != -1:  # Make sure a tab character was found
                content = line_content[tab_index+1:]  # Extract content starting after the tab character
                if line_content.startswith('S-'):
                    s_lines.append(content)
                elif line_content.startswith('T-'):
                    t_lines.append(content)
                elif line_content.startswith('H-'):
                    h_lines.append(content)

# Function to write lines to a CSV file
def write_to_csv(file_name, header, lines):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([header])
        for line in lines:
            writer.writerow([line])

# Write S, T, and H lines to their respective CSV files
write_to_csv('query.csv', 'S', s_lines)
write_to_csv('target.csv', 'T', t_lines)
write_to_csv('output.csv', 'H', h_lines)
