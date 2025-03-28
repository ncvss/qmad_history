with open(f"./test/testresults/dataset_number.txt", "r") as datn:
    dataset_str = datn.read()

print(dataset_str)
dataset_num = int(dataset_str)+1
print(dataset_num)

with open(f"./test/testresults/dataset_number.txt", "w") as datn:
    datn.write(f"{dataset_num:03}")
