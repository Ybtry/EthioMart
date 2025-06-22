import spacy
import os


MODEL_PATH_BEST = "models/model-best"
MODEL_PATH_LAST = "models/model-last"

if os.path.exists(MODEL_PATH_BEST):
    MODEL_TO_LOAD = MODEL_PATH_BEST
elif os.path.exists(MODEL_PATH_LAST):
    MODEL_TO_LOAD = MODEL_PATH_LAST
else:
    print("Error: No trained model found in the 'models/' directory.")
    print("Please ensure your training completed successfully and check the 'models/' folder.")
    exit()
    
print(f"Loading model from: {MODEL_TO_LOAD}")
try:
    nlp = spacy.load(MODEL_TO_LOAD)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you are in the correct virtual environment and the model path is valid.")
    exit()

# Example texts to test
texts_to_test = [
    "ዋጋ:-6800ብር",
    "BARDEFU 2 IN 1 Multi purpose juicer",
    "አድራሻ ቁ.1 👉 መገናኛ ታሜ ጋስ ህንፃ ጎን ስሪ ኤም ሲቲ ሞል ሁለተኛ ፎቅ ቢሮ ቁ. SL-05A(ከ ሊፍቱ ፊት ለ ፊት)",
    "👉ለቡ መዳህኒዓለም ቤተ/ክርስቲያን ወደ ሙዚቃ ቤት ከፍ ብሎ #ዛም_ሞል 2ኛ ፎቅ ቢሮ.ቁ 214",
    "አዲስ ስልክ iPhone 15 Pro Max ዋጋ 85000 ብር አድራሻ ፒያሳ",
    "በጣም የሚያምር ቲሸርት በ500 ብር ብቻ ይግዙ አሁን!",
    "የሚሸጥ መኪና ዋጋ 2 ሚሊዮን ብር",
    "ልዩ ቅናሽ ዛሬ ብቻ! ቦሌ መድሐኔዓለም ግሮሰሪ",
    "ምርጥ ቡና በዋጋ 150 ብር በኪሎ",
    "ለቤት የሚሆን አሪፍ እቃ",
    "ከቦሌ ወደ ቂርቆስ ይሄዳል",
    "በ1000 ብር ብቻ ድንቅ ስራ!"
]

print("\n--- Testing Model with Predefined Texts ---")
for i, text in enumerate(texts_to_test):
    doc = nlp(text)
    print(f"\nText {i+1}: '{text}'")
    if doc.ents:
        for ent in doc.ents:
            print(f"  - Entity: '{ent.text}' | Label: '{ent.label_}' | Span: ({ent.start_char}, {ent.end_char})")
    else:
        print("  - No entities found.")

print("\n--- Testing Model with User Input ---")
while True:
    user_input = input("\nEnter text to extract entities (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    doc = nlp(user_input)
    if doc.ents:
        for ent in doc.ents:
            print(f"  - Entity: '{ent.text}' | Label: '{ent.label_}' | Span: ({ent.start_char}, {ent.end_char})")
    else:
        print("  - No entities found.")

print("\nExiting test script.")