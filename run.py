import json



def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{json_path}'")
    except PermissionError:
        raise PermissionError(f"Permission denied: '{json_path}'")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file '{json_path}': {e}")
def main():
    try:
        data = load_json(data.json)
        print("Data Successfully Loaded.")
    except Exception as e:
        print(f"Error:{e}")
