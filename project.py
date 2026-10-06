import requests
HF_API_KEY="api_key"

model="facebook/bart-large-mnli"
api_url=f"https://router.huggingface.co/hf-inference/models/{model}"
headers={"Authorization": f"Bearer {HF_API_KEY}"}
labels=["SPAM :(", "SAFE :)"]

def classify_message(message):
    payload = {"inputs":message, "parameters":{"candidate_labels":labels}}
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)

    if not response.ok:
        raise RuntimeError(f"API Error:{response.status_code}.")

    data = response.json()
    print(data)
    # results = list(zip(data["labels"], data["scores"]))
    results = [(item["label"], item["score"]) for item in data]

    return sorted(results, key=lambda x: x[1], reverse=True)

def show_results(message, results):
    label,score=results[0]

    print("\n"+"="*44)
    print("Spam vs Safe Message Classifier")
    print("*"*44)
    print(f"Message:{message}")
    print(f"Result: {label} -> Percentage:{score*100:.1f}%\n")

    print("Confidence scores:")

    for i, (LABEL, SCORE) in enumerate(results,1):
        print(f"{i}. {LABEL}: {SCORE*100:.1f}%")

    if label == "Spam":
        print("\n WARNING!!! DO NOT CLICK LINKS OR SHARE PERSONAL INFO!")

    else:
        print("\n LOOKS SAFE!!! Nonetheless, always stay alert!")

    print("=*="*44)

def main():
    print("-----Spam vs Safe Message Classifier Bot -----")
    print("Type the word 'exit to quit\n")

    while True:
        msg = input("Enter message here---->").strip()

        if msg.lower()=="exit":
            print("Goodbye!")
            break
        if not msg:
            print("Please enter your message---->\n")
            continue
        try:
            results = classify_message(msg)
            show_results(msg, results)

        except Exception as e:
            print(f"\n Error: {e}")
            print("Check your API key and internet connection.\n")

if __name__ == "__main__":
    main()
