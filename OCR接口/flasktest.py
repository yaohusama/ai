# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
from extraction_info import orc_text
from flask import jsonify


app = Flask(__name__)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(r'D:\code',secure_filename(f.filename))#安全获取
        print(upload_path)
        # upload_path()
        f.save(upload_path)
        result = orc_text(upload_path)
        return ':'.join(result)
    # if request.method == 'POST':
    #     f = request.files['file']
    #     basepath = path.abspath(path.dirname(__file__))  # 获取当前文件的绝对路径
    #     filename = secure_filename(f.filename)
    #     upload_path = path.join(basepath, 'static', 'uploads', filename)  # 文件要存放的目标位置
    #     f.save(upload_path)
    #     return redirect(url_for('upload'))

    return render_template(r'upload.html')#upload.html必须放在templates文件夹下面

@app.route('/test', methods=['POST', 'GET'])
def testjson():
    return jsonify(name='zhangsan',age=22)

if __name__ == '__main__':
    app.run()