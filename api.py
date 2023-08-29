from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
# from pipline import classifyPose as cp 
from Pose_classification import video 


app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'


cors = CORS(app, resources={r"/vision_iot/*": {"origins": "*"}})
cors = CORS(app, resources={r"/checkservice/*": {"origins": "*"}})


@app.route("/")
def beging():
    return """
    <html> 
        <body>
                <h1> API ERGONOMIA MAOS  </h1>
                    <u1>
                        <a href= "/vision"> Iniciar </a>

                </body> 
    
    </html> """

@app.route('/vision', methods=['GET'])
def vision():
    angles = video()
   
    print(f"Angulos estimados:{angles}")

    l_w = angles[0]
    r_w = angles[1]
    l_i = angles[2]
    r_i = angles[3]
    l_p = angles[4]
    r_p = angles[5]

  
    return  u"""<h2>Angulo punho esquerdo - %s</h2>
                <h2>Angulo punho direito -  %s</h2>
                <h2>Angulo indicador esquerdo - %s</h2>
                <h2>Angulo indicador direito - %s</h2>
                <h2>Angulo minguin esquerdo - %s</h2>
                <h2>Angulo minguin direito - %s</h2>
                
                <a href="/imagem">Voltar</a> 
             """% (l_w, r_w, l_i, r_i, l_p, r_p)

@app.route("/imagem")
def imagem():
    return """
    <html> 
        <body>
                <h1> API ERGONOMIA MAOS  </h1>
                    <u1>
                        <a href= "/vision_iot"> Iniciar novamente </a>

                </body> 
    
    </html> """


@app.route('/vision_iot', methods=['GET','POST'])
def vision_iot():
    angles = video()
    # angles = cp()

    l_w = angles[0]
    r_w = angles[1]
    l_i = angles[2]
    r_i = angles[3]
    l_p = angles[4]
    r_p = angles[5]


    return jsonify(
        Ang_Mao_D = l_w,
        Ang_Mao_E = r_w,
        Ang_Dedo_Ind_D = l_i,
        Ang_Dedo_Ind_E = r_i,
        Ang_Dedo_Peq_D = l_p,
        Ang_Dedo_Peq_E = r_p,
        )



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug =True)