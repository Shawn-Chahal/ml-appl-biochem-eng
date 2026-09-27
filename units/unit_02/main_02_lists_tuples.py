# ============================================================================
#   Lists and Tuples
# ============================================================================
#
#   * Experimental measurements and process variables are often stored as
#     collections of values.
#
#   * Python lists provide a flexible way to organize, access, and modify
#     collections of data using indexing and slicing operations.
#
#   * Understanding lists and tuples is an important step toward working
#     with NumPy arrays, datasets, and machine learning features.
#
# ============================================================================

print()

print("You can make a list of values:")
x = [0, 1, 2, 3, 4]
y = [5, 6, 7, 8, 9, 10, 11]
print(f"  {x = }")
print(f"  {y = }")
print()

print("You can concatenate lists (i.e., join them) by adding them:")
z = x + y
print(f"  {z = }")
print()

print("You can access elements in a list via their index:")
print(f"  {z[0] = }")  # Access the 1st value in the list.
print(f"  {z[1] = }")  # Access the 2nd value in the list.
print(f"  {z[-1] = }")  # Access the last value in the list.
print(f"  {z[-2] = }")  # Access the 2nd last value in the list.
print()

print("You can access a slice of a list by various methods:")
print(f"  {z[2:4] = }")  # Start at idx=2, stop before idx=4.
print(f"  {z[1:10:2] = }")  # Start at idx=1, stop before idx=10, in steps of 2.
print(f"  {z[::-1] = }")  # All indices, in steps of -1. This reverses the list.
print()

print("You can get the length of a list using the len function:")
print(f"  {len(z) = }")
print()

print("You can modify an element in a list:")
z[5] = 473
print(f"  {z = }")
print()

print("You can append a new value to the end of the list:")
z.append(241)
print(f"  {z = }")
print()

print("You can extend a list with values from another list:")
a = [700, 800, 900]
z.extend(a)
print(f"  {z = }")
print()

print("You can make a 2D list by simply creating a \"list of lists\".\n"
      "The same indexing rules apply:")
b = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
]
print(f"  {b = }")
print()

print("Get the list at index 1:")
print(f"  {b[1] = }")
print()

print(f"In the list at index 1, get the value at index 2:")
print(f"  {b[1][2] = }")
print()

print("Note the difference between assignment and copying a list:")
u = [1, 2, 3]
v = u
w = u.copy()
print(f"  {u = }")
print(f"  {v = }  (v and u both refer to the same list in memory.)")
print(f"  {w = }  (w and u are different lists with the same contents.)")
print()

print("Now if you modify the original list:")
u[0] = 9
print(f"  {u = }")
print(f"  {v = }  (changed because v and u refer to the same list)")
print(f"  {w = }  (unchanged because w and u are different lists)")
print()

print("A tuple is a type of immutable list, meaning it is a list that \n"
      "can't be modified:")
x_list = [0, 1, 2, 3, 4]
x_tuple = (0, 1, 2, 3, 4)
print(f"  {x_list = }")
print(f"  {x_tuple = }")
print()

print(f"Try modifying the tuple as you would a list and see what happens...")
print()
