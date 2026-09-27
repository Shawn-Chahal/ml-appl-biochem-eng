# ============================================================================
#   Conditional Statements
# ============================================================================
#
#   * Bioprocesses often require monitoring of operating conditions such as
#     pH and temperature.
#
#   * Conditional statements allow a program to evaluate process conditions
#     and determine whether corrective actions are required.
#
#   * Logical operators (e.g., and, or, not) can be used to combine
#     multiple conditions into more complex decision rules.
#
# ============================================================================

# Constants are typically written in UPPER_CASE.
PH_MIN = 6.5
PH_MAX = 7.5

TEMP_MIN = 36  # C
TEMP_MAX = 38  # C

# Variables are typically written in lower_case.
ph = 8.05
temperature = 35.0

print()

print("Bioreactor Status")
print(f"pH: {ph}")
print(f"Temperature: {temperature} C")
print()

if ph > PH_MAX:
    print("pH is too high.")
    print("Action: Inject acid.")

elif ph < PH_MIN:
    print("pH is too low.")
    print("Action: Inject base.")

else:
    print("pH is within optimal range.")

print()

if temperature > TEMP_MAX:
    print("Temperature is too high.")
    print("Action: Increase cooling water flow rate.")

elif temperature < TEMP_MIN:
    print("Temperature is too low.")
    print("Action: Decrease cooling water flow rate.")

else:
    print("Temperature is within optimal range.")

print()

# You can also assign conditions to variables.
# This can make your code easier to read.
ph_ok = (PH_MIN <= ph <= PH_MAX)
temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)

# You can use "and" when both conditions must be True.
if ph_ok and temp_ok:
    print("Process operating normally.")

else:
    print("Process alarm triggered.")

# You can use 'or' when at least one condition must be True.
if ph_ok or temp_ok:
    print("At least one variable is within optimal range.")

# You can use 'not' to reverse the condition .
# i.e., True becomes False. False becomes True.
if not ph_ok:
    print("pH requires corrective action.")

if not temp_ok:
    print("Temperature requires corrective action.")
