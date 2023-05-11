import json
import csv

jsonl_file_path = "/staging/pandotra/t5_large_Structured_Walmart-Amazon/output_t5/output_Structured_Walmart-Amazon-test.jsonl"
csv_file_path = "/staging/pandotra/t5_large_Structured_Walmart-Amazon/output_t5/output_Structured_Walmart-Amazon-test.csv"

# Open the JSONL file for reading and the CSV file for writing
with open(jsonl_file_path, "r") as jsonl_file, open(csv_file_path, "w", newline="") as csv_file:
    # Create a CSV writer object
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(["left", "right", "match","targets"])
    #writer.writerow(["left", "right", "match","targets","match_confidence"])

    # Read each line in the JSONL file
    for line in jsonl_file:
        # Parse the JSON object from the line
        obj = json.loads(line)

        # Write the fields to the CSV file
        writer.writerow([obj["left"], obj["right"], obj["match"],obj["targets"]])
        #writer.writerow([obj["left"], obj["right"], obj["match"],obj["targets"],obj["match_confidence"]])
