from flask import Flask, request, jsonify
from inference_vietrag7b import generate
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
    question = data.get('question')
    context = data.get('context')

    if not question:
        return jsonify({'error': 'Missing question in body'}), 400
    if not context:
        return jsonify({'error': 'Missing context in body'}), 400
    try:
        answer = generate(question=question, context=context)
    except:
        answer = 'Error when call model inference. It may be cause by context too long lead to out of memory'

    # Xử lý dữ liệu và trả lại kết quả (trong ví dụ này là chỉ trả về lại dữ liệu nhận được)
    response = {
        'answer': answer,
    }
    return jsonify(response), 200

# Chạy Flask app
if __name__ == '__main__':
    app.run(port=6000, debug=False)
