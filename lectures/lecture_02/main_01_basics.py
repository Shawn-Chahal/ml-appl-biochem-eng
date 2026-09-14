# ============================================================================
#   Python Fundamentals
# ============================================================================
#
#   * Variables are used to store experimental data, process parameters, and
#     machine learning inputs and outputs.
#
#   * Common data types include booleans, integers, floats, and strings.
#
#   * These foundational concepts will be used throughout the course for data
#     analysis, visualization, and machine learning applications.
#
# ============================================================================

print()

print("The print function can be used to print information from our code to the terminal.")
print("* E.g., print('Hello world!')")
print()

print("When you use the # symbol, everything on that line after the # becomes a comment and will not execute.")
print("* Comment starts after print operation, therefore this will execute.")  # This is a comment
# print("Comment starts before the print operation, therefore this will not execute.")
print()

print("A boolean can have the value True or False.")
x_bool = True
print(f"* E.g., {x_bool = }")
print()

print("An integer is a whole number and can be positive or negative.")
x_int = 1
print(f"* E.g., {x_int = }")
print()

print("A float is a decimal number and can be positive or negative.")
x_float = -2.3678
print(f"* E.g., {x_float = }")
print()

print("A string is a sequence of characters.")
x_string = "Hello!"
print(f"* E.g., {x_string = }")
print()

print("You can place multiple variables within an f-string:")
p_scale = "250 mL"
p_type = "fed-batch"
c_glucose = 21  # mM (It's a good idea to write the units of any values as a comment so you don't lose track of it.)
print(f"* The glucose concentration in the {p_scale} {p_type} process is {c_glucose} mM.")
print()

print("Note the difference between an f-string and a self-documenting expression:")
x = 5
y = 2
print(f"* f-string: ({x}, {y})")
print(f"* self-documenting expression: ({x=}, {y=})")
print()

print("You can perform a variety of mathematical operations:")
z_add = x + y  # Addition
z_sub = x - y  # Subtraction
z_mul = x * y  # Multiplication
z_div = x / y  # Division
z_pow = x ** y  # Power
z_mod = x % y  # Modulo
print(f"  {x = }")
print(f"  {y = }")
print(f"  {z_add = }")
print(f"  {z_sub = }")
print(f"  {z_mul = }")
print(f"  {z_div = }")
print(f"  {z_pow = }")
print(f"  {z_mod = }")
print()
