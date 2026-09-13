##############################################################################
#   You can define a class to represent a system.                            #
#   For example, we can create a Bioreactor class to simplify our previous   #
#   process control example to simulate different bioprocesses easily.       #
#   Class names are typically written in CapitalizedWords.                   #
##############################################################################

# CLASSES

class Bioreactor:
    def __init__(self, process_name, ph_min, ph_max, ph_change, temp_min, temp_max, temp_change):
        self.process_name = process_name
        self.ph_min = ph_min
        self.ph_max = ph_max
        self.ph_change = ph_change
        self.temp_min = temp_min
        self.temp_max = temp_max
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
        print(f"\n===== {self.process_name} START =====")
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

        print(f"\n===== {self.process_name} END =======")
        print()


# MAIN PROGRAM

# Instantiate bioreactor objects from the Bioreactor class
bioreactor_1 = Bioreactor(process_name="Ethanol fermentation",
                          ph_min=4.0, ph_max=5.0, ph_change=0.1,
                          temp_min=28, temp_max=32, temp_change=0.4)

bioreactor_2 = Bioreactor(process_name="Thermophilic anaerobic digestion",
                          ph_min=6.8, ph_max=7.8, ph_change=0.1,
                          temp_min=50, temp_max=60, temp_change=0.8)

# Run simulations
bioreactor_1.simulate_process(initial_ph=5.15, initial_temperature=26.1)
bioreactor_2.simulate_process(initial_ph=6.35, initial_temperature=62.2)
