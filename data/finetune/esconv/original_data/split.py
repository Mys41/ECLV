import json
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with open('ESConv.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    train_data, test_val_data = train_test_split(data, test_size=0.4)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5)
    with open('train_data.txt', 'w', encoding='utf-8') as f:
        json.dump(train_data, f)

    with open('val_data.txt', 'w', encoding='utf-8') as f:
        json.dump(val_data, f)

    with open('test_data.txt', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)