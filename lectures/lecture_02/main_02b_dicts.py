# ============================================================================
#   Dictionaries
# ============================================================================
#
#   * A dictionary is a unique way of holding a collection of data.
#
#   * It is not simply an ordered list indexed by 0, 1, 2, 3, etc.
#
#   * Instead we can define our own keys to access different values.
#
# ============================================================================

print()

print("You can make a dictionary using keys and values:")
x_dict_1 = dict(mode="batch", volume="3 L", initial_glucose=30, initial_pH=7.5)
print(f"  {x_dict_1 = }")
print()

print("When using the dict(key1=value1, key2=value2, etc.) syntax, the keys become strings.")
print("For more flexibility when making keys, you can use the following syntax:")
x_dict_2 = {3: "Value 1", (30, 7.5): True, "mode": "batch", "PID_active": ["T", "pH", "DO"]}
print(f"  {x_dict_2 = }")
print()

print("Although the syntax is different, they are both dictionaries:")
print(f"  {type(x_dict_1) = }")
print(f"  {type(x_dict_2) = }")
print()

print("Note: You can't use a list as a key, but you can use a tuple.")
print("      Both lists and tuples can be a value though.")
print()

print("You can access a value from a dict using the appropriate key:")
print(f"  {x_dict_1['mode'] = }")
print(f"  {x_dict_1['initial_glucose'] = }")
print(f"  {x_dict_2['mode'] = }")
print(f"  {x_dict_2[3] = }")
print(f"  {x_dict_2[(30, 7.5)] = }")
print(f"  {x_dict_2["PID_active"] = }")
print()

print("If a dict value is a list, you can chain indices together:")
print(f"  {x_dict_2["PID_active"][0] = }")
print(f"  {x_dict_2["PID_active"][1] = }")
print(f"  {x_dict_2["PID_active"][2] = }")
print()

print("You can easily add new key-value pairs:")
x_dict_1["mixing_speed"] = "100 RPM"
print(f"  {x_dict_1 = }")
print()

print("You can't repeat a key in the same dictionary,")
print("it will just overwrite the previous value:")
x_dict_1["initial_glucose"] = 7.5
print(f"  {x_dict_1 = }")
print()

print("Note: Values can be repeated in a dictionary.")
print()

print("You can get all the keys and values in a dictionary:")
print(f"  {x_dict_1.keys() = }")
print(f"  {x_dict_1.values() = }")
print(f"  {x_dict_1.items() = }")
print()
