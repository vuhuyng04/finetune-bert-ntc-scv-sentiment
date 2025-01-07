from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn tới checkpoint
CHECKPOINT_DIR = r"model/best_model"

# Tải mô hình và tokenizer từ checkpoint
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

# Mảng ánh xạ kết quả dự đoán
labels = ["Negative", "Positive"]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Lấy dữ liệu từ form
        sentence = request.form.get("sentence")
        
        if not sentence:
            return render_template("index.html", result="No sentence provided")

        # Tokenize câu đầu vào
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        # Dự đoán
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=1).item()

        # Ánh xạ dự đoán thành nhãn "Negative" hoặc "Positive"
        result = labels[prediction]

        return render_template("index.html", sentence=sentence, result=result)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
