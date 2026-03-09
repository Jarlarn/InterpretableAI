def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                labels.append(int(parts[0]))
                texts.append(parts[1])
    return texts, labels


train_texts, train_labels = load_data("data/ReviewBaseTraining.txt")
val_texts, val_labels = load_data("data/ReviewBaseValidation.txt")
test_texts, test_labels = load_data("data/ReviewBaseTest.txt")


print(test_texts[0])
for i, text in enumerate(test_texts):
    if len(text.split()) < 30:
        print(i, text)
