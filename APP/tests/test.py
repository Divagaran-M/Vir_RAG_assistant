from services.extractor import extract_text

print("Starting extraction...")

text = extract_text("data/uploads/605db9b4cd7041178cce3d127aab7681_Divagaran_Resume.pdf")

print("Extraction finished!")

print("Length:", len(text))
print(text[:500])