import re

def print_keys_values(obj):
    if isinstance(obj, dict):
        # If it's a dictionary
        for key, value in obj.items():
            print(f"{key}: {value}")
    elif hasattr(obj, '__dict__'):
        # If it's an instance of a class
        for key, value in vars(obj).items():
            print(f"{key}: {value}")
    else:
        print("Unsupported object type")

def print_attribute_from_instances(instances, attribute_name):
    for instance in instances:
        # Use getattr to access the attribute dynamically
        attribute_value = getattr(instance, attribute_name, None)
        if attribute_value is not None:
            print(f"{attribute_name} for {instance.__class__.__name__}: {attribute_value}")
        else:
            print(f"{attribute_name} not found for {instance.__class__.__name__}")

def print_many_attributes_from_instances(instances, attribute_names):
    # print(f"\nAttributes for {instance.__class__.__name__}:")
    for ix, instance in enumerate(instances):
        for attribute_name in attribute_names:
            # Use getattr to access the attribute dynamically
            attribute_value = getattr(instance, attribute_name, None)
            if attribute_value is not None:
                print(f"{ix}: {attribute_name}: {attribute_value}")
            else:
                print(f"{ix}: {attribute_name} not found")