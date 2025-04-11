import json

jsonpath1 = 'E:\Projects\detr\\base_ds_coco.json'
jsonpath2 = 'E:\Projects\dataset\coco\\annotations\instances_val2017.json'

def compare_json_files(file1, file2):
    try:
        # Load the first JSON file
        with open(file1, 'r') as f1:
            data1 = json.load(f1)
        
        # Load the second JSON file
        with open(file2, 'r') as f2:
            data2 = json.load(f2)
        
        # Compare the two JSON objects
        if data1 == data2:
            print("The two JSON files are the same.")
        else:
            print("The two JSON files are different.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to compare the files
compare_json_files(jsonpath1, jsonpath2)

