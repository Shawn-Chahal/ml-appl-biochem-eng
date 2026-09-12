PH_MIN = 6.5
PH_MAX = 7.5

TEMP_MIN = 36  # C
TEMP_MAX = 38  # C

ph = 8.0
temperature = 35

ph_ok = (PH_MIN <= ph <= PH_MAX)
temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
process_ok = (ph_ok and temp_ok)
print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.1f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok})")

# We can use a while loop to run code "while" a condition is True
while not process_ok:

    if ph > PH_MAX:
        print("Injecting acid...")
        ph -= 0.1  # Subtracts 0.1 from the value of ph
    elif ph < PH_MIN:
        print("Injecting base...")
        ph += 0.1  # Adds 0.1 to the value of ph

    if temperature > TEMP_MAX:
        print("Increasing cooling water flow rate...")
        temperature -= 0.3  # Subtracts 0.2 from the value of temperature
    elif temperature < TEMP_MIN:
        print("Decreasing cooling water flow rate...")
        temperature += 0.3  # Adds 0.2 to the value of temperature

    ph_ok = (PH_MIN <= ph <= PH_MAX)
    temp_ok = (TEMP_MIN <= temperature <= TEMP_MAX)
    process_ok = (ph_ok and temp_ok)
    print(f"\nBioreactor status ({process_ok = }) | pH: {ph:.1f} ({ph_ok = }) | T: {temperature:.1f} C ({temp_ok = })")
