from flask import Flask, request, jsonify
from inference_llama import generate
# from inference_vietrag7b import generate as vietrag_generate


app = Flask(__name__)

# API endpoint sử dụng phương thức POST
@app.route('/generate', methods=['POST'])
def handle_post():
    # Lấy dữ liệu JSON từ request body
    data = request.get_json()

    # Kiểm tra xem dữ liệu có hợp lệ không
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Giả sử chúng ta chỉ muốn nhận và trả về một số trường trong JSON
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt and model are required'}), 400
    answer = ''
    answer = generate(prompt=prompt)

    # Xử lý dữ liệu và trả lại kết quả (trong ví dụ này là chỉ trả về lại dữ liệu nhận được)
    response = {
        'answer': answer,
    }
    return jsonify(response), 200

# Chạy Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=False)
