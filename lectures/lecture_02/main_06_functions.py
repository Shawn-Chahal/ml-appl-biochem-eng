#########################################################################
#   You can define a function to perform different tasks.               #
#   Functions make it is easy to reuse code.                            #
#   It also makes it clear what a piece of code is doing.               #
#   For example, these functions can be used to simplify our previous   #
#   process control example to run multiple simulations easily.         #
#   Function names are typically written in lower_case.                 #
#########################################################################

# CONSTANTS

PH_MIN = 6.5
PH_MAX = 7.5

TEMP_MIN = 36  # C
TEMP_MAX = 38  # C


# FUNCTIONS

def in_optimal_range(ph, temperature):
    ph_ok = (PH_MIN <= ph <= PH_MAX)
    temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
    process_ok = (ph_ok and temp_ok)
    print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.1f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok})")

    return process_ok


def simulate_process(ph, temperature):
    while not in_optimal_range(ph, temperature):
        if ph > PH_MAX:
            print("Injecting acid...")
            ph -= 0.1
        elif ph < PH_MIN:
            print("Injecting base...")
            ph += 0.1

        if temperature > TEMP_MAX:
            print("Increasing cooling water flow rate...")
            temperature -= 0.3
        elif temperature < TEMP_MIN:
            print("Decreasing cooling water flow rate...")
            temperature += 0.3


# MAIN PROGRAM

simulation_id = 0
for ph_val in [6, 7, 8]:
    for temp_val in [35, 37, 39]:
        simulation_id += 1
        print(f"\n\n========== Simulation {simulation_id} ==========")
        simulate_process(ph_val, temp_val)
