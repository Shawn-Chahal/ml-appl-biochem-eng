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

# ============================================================================
#   PROBLEM 
# ============================================================================
#
#   Consider the following dataset with:
#    * 5 samples
#    * 3 features
#       - Concentration of compound A [mM]
#       - Concentration of compound B [mM]
#       - Concentration of compound C [mM]
#
#   -------------------------------
#   | sample_id | C_A | C_B | C_C |
#   -------------------------------
#   | sample_0  | 1.5 | 2.9 | 3.5 |
#   | sample_1  | 5.8 | 2.7 | 9.2 |
#   | sample_2  | 6.2 | 2.4 | 8.4 |
#   | sample_3  | 2.9 | 9.3 | 6.6 |
#   | sample_4  | 2.1 | 6.6 | 5.1 |
#   -------------------------------
#
#   Calculate the mean and standard deviation of each compound's
#   concentration (i.e., C_A, C_B, C_C).
#
# ============================================================================

# ============================================================================
#   SOLUTION 
# ============================================================================
#
#   Approach:
#     * We will create a 2D numpy array to store this data.
#     * Each row will represent a sample.
#     * Each column will represent the concentration of a compound.
#     * This will result in a 5 x 3 array.
#     * We will then use numpy functions to calculate the mean and SD.
#
#   Note:
#     * An array can have more than 2 dimensions (or axis) as well.
#     * Rows:    axis=0
#     * Columns: axis=1
#     * etc...:  axis=2, 3, 4, 5
#
# ============================================================================

print("============\n  SOLUTION  \n============\n")

print("For each sample we can make a list of concentrations:")
sample_0 = [1.5, 2.9, 3.5]
sample_1 = [5.8, 2.7, 9.2]
sample_2 = [6.2, 2.4, 8.4]
sample_3 = [2.9, 9.3, 6.6]
sample_4 = [2.1, 6.6, 5.1]
print("              C_A  C_B  C_C")
print(f"  {sample_0 = }")
print(f"  {sample_1 = }")
print(f"  {sample_2 = }")
print(f"  {sample_3 = }")
print(f"  {sample_4 = }")
print()

print("We can then stack these lists in a 2D list: ")
x = [
    sample_0,
    sample_1,
    sample_2,
    sample_3,
    sample_4
]
print(f"  {x = }")
print()
print("We can then convert the 2D list into a 2D array: ")
x = np.array(x)
print(f"  {x = }")
print()

print(f"Note the slight difference in syntax between accessing values \n"
      f"from a 2D numpy array vs 2D list:")
print(f"  * Numpy syntax: {x[1, 2] = }")
print(f"  * Lists syntax: {x[1][2] = }")
print()

# ============================================================================
#
#   By default, numpy will calculate the mean and SD across all elements in
#   an array. However, we can also specify which axis to calculate these
#   stats across. Think carefully about out goal: For each feature (i.e.,
#   column) we want to calculate the mean and SD across samples (i.e. rows,
#   a.k.a. axis=0).
#
#   Another way of thinking about it is that the different values we are
#   using to calculate the mean and SD appear in different rows. Rows are
#   represented in axis=0, therefore we want to calculate the mean and SD
#   across axis=0.
#
#   A third way of thinking about it is that axis=0 will be "collapsed" or
#   "aggregated". Our numpy array "x" has a shape of (5, 3). We want x_mean
#   and x_sd to have a shape of (3, ) or (1, 3). This is done by collapsing
#   or aggregating axis=0.
#
# ============================================================================

print("We can now calculate the mean and SD of our 2D array across axis=0:")
x_mean = np.mean(x, axis=0)
x_sd = np.std(x, axis=0)
print(f"  {x_mean = }")
print(f"  {x_sd = }")
print()

print("We can also see the shape of numpy arrays:")
print(f"  {x.shape = }")
print(f"  {x_mean.shape = }")
print(f"  {x_sd.shape = }")
print()

# ============================================================================
#
#   We can nicely format our final answer using f-strings. For example, we
#   can set the number of decimals to be printed by using the ".2f" to
#   indicate 2 decimal places. This will only affect the appearance of the
#   value in the string when printing, it won't change the original value of
#   the variable in your code.
#
# ============================================================================

print("ANSWER:")
print(f"  Concentration A: {x_mean[0]:.2f} +/- {x_sd[0]:.2f} mM")
print(f"  Concentration B: {x_mean[1]:.2f} +/- {x_sd[1]:.2f} mM")
print(f"  Concentration C: {x_mean[2]:.2f} +/- {x_sd[2]:.2f} mM")
