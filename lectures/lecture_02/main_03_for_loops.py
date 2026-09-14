# ============================================================================
#   For Loops and List Comprehensions
# ============================================================================
#
#   * Many machine learning and data analysis tasks require processing large
#     collections of samples, features, or experimental measurements.
#
#   * For loops provide a way to systematically iterate through data and
#     perform repeated calculations.
#
#   * List comprehensions offer a compact and efficient method for generating
#     new datasets and transformed features.
#
# ============================================================================

print()

print("You can use a for loop to iterate over a range of numbers:")
for i in range(10):  # Iterate 10 times ranging from 0 to 9
    print(f"  {i = }")

print()

print("You can iterate over elements in a list directly:")
process_types = ["Batch", "Fed-Batch", "Continuous"]
for process_type in process_types:
    print(f"  * {process_type}")

print()

print("You can keep track of a number as you iterate over a list:")
for i, process_type in enumerate(process_types, start=1):
    print(f"  {i}) {process_type}")

print()

print("You can use a list comprehension to automate list construction:")
x = [i ** 2 for i in range(10)]  # Make a list with the first 10 squares
print(f"  {x = }")

print()
