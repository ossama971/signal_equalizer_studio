import csv

def transform_rows_to_columns(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        rows = list(reader)

        if len(rows) != 2:
            print("Error: Input file should have exactly two rows.")
            return

        data_transposed = zip(rows[0], rows[1])  # Combining rows into two columns

        for row in data_transposed:
            writer.writerow(row)

    print(f"Data transformed into two columns and saved to '{output_file}'")

# Replace 'input.csv' and 'output.csv' with your file names
transform_rows_to_columns('Abnormality 1.csv', 'output.csv')
