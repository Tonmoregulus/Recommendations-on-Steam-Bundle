import pickle
import json

file_path = 'bundle_item_map'

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    print(data)

output_file = "bundle_item.json"

with open(output_file, 'w') as f:
    for key, value in data.items():
        json_line = json.dumps({key: list(value)}) 
        f.write(json_line + '\n') 

print(f"数据已成功写入 {output_file}")