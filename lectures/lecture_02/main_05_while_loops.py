PH_MIN = 6.5
PH_MAX = 7.5
PH_CHANGE = 0.1

TEMP_MIN = 36  # C
TEMP_MAX = 38  # C
TEMP_CHANGE = 0.3

ph = 8.05
temperature = 35.0

ph_ok = (PH_MIN <= ph <= PH_MAX)
temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
process_ok = (ph_ok and temp_ok)
print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.2f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok = })")

# We can use a while loop to run code "while" a condition is True
while not process_ok:

    if ph > PH_MAX:
        print("Injecting acid...")
        ph -= PH_CHANGE  # Subtracts PH_CHANGE from the value of ph
    elif ph < PH_MIN:
        print("Injecting base...")
        ph += PH_CHANGE  # Adds PH_CHANGE to the value of ph

    if temperature > TEMP_MAX:
        print("Increasing cooling water flow rate...")
        temperature -= TEMP_CHANGE  # Subtracts TEMP_CHANGE from the value of temperature
    elif temperature < TEMP_MIN:
        print("Decreasing cooling water flow rate...")
        temperature += TEMP_CHANGE  # Adds TEMP_CHANGE to the value of temperature

    ph_ok = (PH_MIN <= ph <= PH_MAX)
    temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
    process_ok = (ph_ok and temp_ok)
    print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.2f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok = })")
