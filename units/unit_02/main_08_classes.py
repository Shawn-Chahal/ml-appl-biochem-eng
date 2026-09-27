# ============================================================================
#   Classes
# ============================================================================
#
#   * Classes provide a way to represent systems by combining attributes
#     (data) and methods (behavior).
#
#   * In this example, a Bioreactor object stores process-specific operating
#     limits and provides methods for simulating process control actions.
#
#   * Many machine learning libraries use classes to represent models and
#     their associated functionality.
#
# ============================================================================

# Class names are typically written in CapitalizedWords.
class Bioreactor:
    def __init__(self, process_name, ph_lims, ph_change, temp_lims, temp_change):
        self.process_name = process_name
        self.ph_min = ph_lims[0]
        self.ph_max = ph_lims[1]
        self.ph_change = ph_change
        self.temp_min = temp_lims[0]
        self.temp_max = temp_lims[1]
        self.temp_change = temp_change

    def in_optimal_range(self, ph, temperature):
        ph_ok = (self.ph_min <= ph <= self.ph_max)
        temp_ok = (self.temp_min <= temperature <= self.temp_max)
        process_ok = (ph_ok and temp_ok)
        print(f"\n"
              f"Bioreactor status ({process_ok = }) | "
              f"pH: {ph:.2f} ({ph_ok = }) | "
              f"T: {temperature:.1f} C ({temp_ok = })")

        return process_ok

    def simulate_process(self, initial_ph, initial_temperature):
        print(f"\n===== {self.process_name} CONTROL START =====")
        ph = initial_ph
        temperature = initial_temperature
        process_ok = self.in_optimal_range(ph, temperature)
        while not process_ok:
            if ph > self.ph_max:
                print("Injecting acid...")
                ph -= self.ph_change
            elif ph < self.ph_min:
                print("Injecting base...")
                ph += self.ph_change

            if temperature > self.temp_max:
                print("Increasing cooling water flow rate...")
                temperature -= self.temp_change
            elif temperature < self.temp_min:
                print("Decreasing cooling water flow rate...")
                temperature += self.temp_change

            process_ok = self.in_optimal_range(ph, temperature)

        print(f"\n===== {self.process_name} CONTROL END =======")
        print()


# ============================================================================
#   Main Program
# ============================================================================
#
#   * Create Bioreactor objects from the Bioreactor class.
#
#   * Each object can store different operating limits and process-specific
#     control parameters.
#
#   * Run simulations using the methods defined within the class.
#
# ============================================================================

# Instantiate Bioreactor objects from the Bioreactor class
bioreactor_1 = Bioreactor(process_name="Ethanol fermentation",
                          ph_lims=(4.0, 5.0), ph_change=0.1,
                          temp_lims=(28, 32), temp_change=0.4)

bioreactor_2 = Bioreactor(process_name="Thermophilic anaerobic digestion",
                          ph_lims=(6.8, 7.8), ph_change=0.1,
                          temp_lims=(50, 60), temp_change=0.8)

# Run simulations
bioreactor_1.simulate_process(initial_ph=5.15, initial_temperature=26.1)
bioreactor_2.simulate_process(initial_ph=6.35, initial_temperature=62.2)
