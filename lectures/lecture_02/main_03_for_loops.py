# You can use a for loop to iterate over a range of numbers
for i in range(10):  # Iterate 10 times ranging from 0 to 9
    print(f"{i = }")

print()

# You can also iterate over elements in a list directly
process_types = ["Batch", "Fed-Batch", "Continuous"]
for process_type in process_types:
    print(process_type)

print()

# You can also keep track of a number as you iterate over a list
for i, process_type in enumerate(process_types, start=1):
    print(f"{i}) {process_type}")
