#########################################################################
#   Functions make it is easier to reuse code.                          #
#   They also makes it more clear what a piece of code is doing.        #
#   Function names are typically written in lower_case.                 #
#########################################################################

# CONSTANTS

PH_MIN = 6.5
PH_MAX = 7.5
PH_CHANGE = 0.1

TEMP_MIN = 36  # C
TEMP_MAX = 38  # C
TEMP_CHANGE = 0.3


# FUNCTIONS

def in_optimal_range(ph, temperature):
    ph_ok = (PH_MIN <= ph <= PH_MAX)
    temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
    process_ok = (ph_ok and temp_ok)
    print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.2f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok = })")

    return process_ok


# MAIN PROGRAM

ph = 8.05
temperature = 35.0

process_ok = in_optimal_range(ph, temperature)

while not process_ok:

    if ph > PH_MAX:
        print("Injecting acid...")
        ph -= PH_CHANGE
    elif ph < PH_MIN:
        print("Injecting base...")
        ph += PH_CHANGE

    if temperature > TEMP_MAX:
        print("Increasing cooling water flow rate...")
        temperature -= TEMP_CHANGE
    elif temperature < TEMP_MIN:
        print("Decreasing cooling water flow rate...")
        temperature += TEMP_CHANGE

    process_ok = in_optimal_range(ph, temperature)
