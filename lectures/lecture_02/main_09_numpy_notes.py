# ============================================================================
#   NumPy Arrays
# ============================================================================
#
#   * Machine learning datasets are often organized as NumPy arrays.
#
#   * Rows typically represent samples, while columns represent measured
#     features or process variables.
#
#   * NumPy makes it easy to calculate statistics, perform mathematical
#     operations, and analyze multidimensional datasets.
#
# ============================================================================

# We need to import the numpy library
import numpy as np

# You can create lists
x_list = [7, 4, 5]
y_list = [2, 1, 3]

# You can create numpy arrays from lists
x_array = np.array(x_list)
y_array = np.array(y_list)

print("Numpy arrays are distinct from lists:")
print(f"  {x_list = }")
print(f"  {y_list = }")
print(f"  {x_array = }")
print(f"  {y_array = }")
print()

print("Adding two lists together will concatenate them:")
print(f"  {x_list = }")
print(f"  {y_list = }")
print(f"  {x_list + y_list = }")
print()

print("Adding two numpy arrays will perform element-wise addition:")
print(f"  {x_array = }")
print(f"  {y_array = }")
print(f"  {x_array + y_array = }")
print()

print("You can perform many types of element-wise operations using numpy:")
print(f"  Addition:       {x_array + y_array = }")
print(f"  Subtraction:    {x_array - y_array = }")
print(f"  Multiplication: {x_array * y_array = }")
print(f"  Division:       {x_array / y_array = }")
print(f"  Power:          {x_array ** y_array = }")
print()

print("You can also perform the dot product using numpy:")
print(f"  Dot product:    {np.dot(x_array, y_array) = }")
print()

print("Many common functions are already built in to numpy:")
print(f"  Square root:   {np.sqrt(x_array) = }")
print(f"  Sine function: {np.sin(x_array) = }")
print(f"  Log(Base10):   {np.log10(x_array) = }")
print(f"  Exponential:   {np.exp(x_array) = }")
print()

print("You can obtain basic stats using numpy:")
a = np.array([2, 3, 1, 12, 5])
print(f"  Data:      {a = }")
print(f"  Mean:      {np.mean(a) = }")
print(f"  Std. Dev.: {np.std(a) = }")
print(f"  Median:    {np.median(a) = }")
print(f"  Maximum:   {np.max(a) = }")
print(f"  Minimum:   {np.min(a) = }")
print(f"  Summation: {np.sum(a) = }")
print()

print("You can round the values in a numpy array:")
b = np.array([79 / 3, 89 / 7, 101 / 13, 127 / 31, 163 / 61])
print(f"  {b = }")
print(f"  {np.round(b, 3) = }")
print(f"  {np.round(b, 2) = }")
print(f"  {np.round(b, 1) = }")
print()

print("Some numpy functions can be called directly from the array itself:")
print(f"  {b = }")
print(f"  {b.round(3) = }")
print(f"  {b.round(2) = }")
print(f"  {b.round(1) = }")
print()

print(f"Note that b remains unchanged. The new rounded array must be assigned to a variable:")
b_round = b.round(1)
print(f"  {b = }")
print(f"  {b_round = }")
print()

print("You can extract a sorted array of the unique values in an array:")
c = np.array(["HEK 293", "HEK 293", "E. coli", "S. cerevisiae",
              "HEK 293", "E. coli", "S. cerevisiae", "HEK 293"])
print(f"  {c = }")
print(f"  {np.unique(c) = }")
print()

print("You can create a linearly spaced array of values:")
ph_vals = np.linspace(6, 8, 11)  # Create 11 equally spaced values between (and including) 6 and 8.
print(f"  {ph_vals = }")
print()

print("You can create a mask to select specific values from an array:")
mask_acidic = ph_vals < 7.0
print(f"  {mask_acidic = }")
print(f"  {ph_vals[mask_acidic] = }")
print()

print("You can invert masks as well:")
mask_basic_or_neutral = ~mask_acidic
print(f"  {mask_basic_or_neutral = }")
print(f"  {ph_vals[mask_basic_or_neutral] = }")
print()

print("You can add masks together if you want to select values where EITHER condition is True:")
ph_lims = (7.1, 7.5)
mask_ph_low = ph_vals < ph_lims[0]
mask_ph_high = ph_vals > ph_lims[1]
mask_ph_not_ok = mask_ph_low + mask_ph_high
print(f"  {mask_ph_not_ok = }")
print(f"  {ph_vals[mask_ph_not_ok] = }")
print()

print("You can multiply masks together if you want to select values where BOTH conditions are True:")
mask_ph_ok = (ph_vals >= ph_lims[0]) * (ph_vals <= ph_lims[1])
print(f"  {mask_ph_ok = }")
print(f"  {ph_vals[mask_ph_ok] = }")
print()
