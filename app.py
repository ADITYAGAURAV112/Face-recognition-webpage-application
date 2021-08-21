import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, flash, request, redirect, url_for,render_template,send_file, jsonify,session
from flask.wrappers import Response
from werkzeug.utils import secure_filename #send_file, send_from_directory
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import pymongo, random,string
import cv2,torch,time
import numpy as np
import base64
import io
from datetime import datetime
import uuid
from collections import defaultdict

session_fold = "static/session"
IMAGE_FOLDER = 'static/photos'
pt_file = "static/data.pt"
#PICKLE_FILE = 'static/embeded_faces.pkl'
if os.path.exists(IMAGE_FOLDER) is False:
    os.mkdir(IMAGE_FOLDER)
ALLOWED_EXTENSIONS = {'jpg', 'png'}
app = Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config["session_fold"] = session_fold
app.secret_key = 'super secret key'
# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["TPVISION"]
mycol = mydb["personemp"]

framelist = mydb["framelist"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uniquify_filename(fil_name):
    fil_name = secure_filename(fil_name)
    fn, ext = os.path.splitext(fil_name)

    fil_name = fn + "_" + "".join(random.sample(string.ascii_lowercase, 8)) + ext
    cursor = mycol.count_documents({'unique_filename': fil_name})
    while cursor > 0:
        fil_name = fn + "_" + random.sample(string.ascii_lowercase, 8) + ext
        cursor = mycol.count_documents({'unique_filename': fil_name})
    return fil_name

def save_data():

    dataset = datasets.ImageFolder("static/photos") # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])       

    #print(name_list)
    data = [embedding_list, name_list] 
    torch.save(data, pt_file) # saving data.pt file

# TODO: handle the case for large number of users,to insure loaded data does not flow RAM
def from_pt():
    load_data = torch.load(pt_file) 
    #embedding_list = load_data[0] 
    #name_list = load_data[1]
    #print(load_data)
    return load_data

def name_to_pt(img,name):
    try:
        load_data = from_pt()
    except:
        save_data()
        load_data = from_pt()
    #load_data = torch.load('data.pt')
    embedding_list = load_data[0]
    name_list = load_data[1]

    # save file 
    img =Image.open(img)
    filepath = 'static/bin/temp.jpg'
    img.save(filepath)
    #im = numpy.fromstring(filepath, numpy.uint8)
    im = cv2.imread(filepath)
    #im =np.array(img)
    #print("--------------------",type(im),type(img))
    image = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    face, prob = mtcnn0(image, return_prob=True)
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(name)
    # save data
    data = [embedding_list, name_list] 
    torch.save(data, pt_file)

# @app.route('/',methods = ['GET','POST'])
# def hello_world():
#     my_id = uuid.uuid1()
#     session["user"] = my_id
#     return render_template("home.html")

@app.route('/image_rec',methods = ["GET",'POST'])
def image_rec():
    if request.method == 'POST':
        if 'act' not in request.files:
            flash ("No file part")
            return redirect(request.url)
        img = request.files['act']
        if img.filename == "":
            flash("NO file selected!!, Please select the file")
            return redirect(request.url)
        try:
            load_data = from_pt()
        except:
            save_data()
            load_data = from_pt()
        embedding_list = load_data[0]
        name_list = load_data[1]
        #print(name_list)
        im = Image.open(img)
        im.save("static/bin/tempo.jpg")
        img = cv2.imread("static/bin/tempo.jpg")
        #img = np.array(im)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
                
            for i, prob in enumerate(prob_list):
                if prob>0.90:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list) # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                    fold = name_list[min_dist_idx] # get name corrosponding to minimum dist
                    #print('fold',fold)
                    try:
                        cursor = mycol.count_documents({'unique_filename':fold})
                        #print("cursor-------------",cursor)
                    except:
                        print("found nothing in monodb")
                        cursor = 0
                        pass
                    if cursor > 0:
                        #print("Returning one file")
                        cursor = mycol.find({'unique_filename':fold })
                        imgrec = cursor[0]
                        #print(imgrec)
                        name = imgrec["user_name"]
                        print("name",name)
                    #else:
                    #    name = ""
                
                    box = boxes[i]
                    if min_dist<0.90:
                        #print('----------------')
                        img = cv2.putText(img, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                
                    img = cv2.rectangle(img, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)

        
        if "user" in session:
            ses = str(session["user"])
        else:
            ses = session['user'] = uuid.uuid1()
            ses = str(ses)
        
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H%M%S%f")
        dt_string = str(dt_string)
        save_path = 'static/session/{}/{}.jpg'.format(ses,dt_string)
        if not os.path.exists(os.path.join(session_fold,ses)):
            os.mkdir(os.path.join(session_fold,ses))
        cv2.imwrite(save_path,img)
        return send_file(save_path)
        # target_filename = 'static/bin/pic.jpg'
        # cv2.imwrite(target_filename,img)
        # return send_file("static/bin/pic.jpg")
    else:
        return render_template("picture.html")




@app.route('/pic_register',methods = ['GET','POST'])
def pic_registration():
    if request.method == "POST":
        if 'love' not in request.files:
            flash("NO file apart")
            return redirect(request.url)
        file = request.files['love']
        if file.filename == "":
            flash("No selected files")
            return redirect(request.url)
        if "Naaam" not in request.form:
            flash("Enter a valid name!!")
            return redirect(request.url)
        text = request.form['Naaam']
        #unique_name = uniquify_filename(file.filename)

        if file and allowed_file(file.filename):
            if not os.path.exists(os.path.join(IMAGE_FOLDER,text)):
                    unique_name = uniquify_filename(text)
                    #unique_name = secure_filename(unique_name)
                    os.mkdir(os.path.join(app.config['IMAGE_FOLDER'],unique_name))
            else:
                    unique_name = text
                #file.save(os.path.join(app.config['IMAGE_FOLDER']),fold)
        else:
            flash("please select '.jpg' format")
            return redirect(request.url)
        
        mycol.insert_one(
                    {
                        'userfilename':file.filename,
                        'unique_filename': unique_name,
                        'user_name':text,
                        'annotated': False
                    }
                    )

        img_name = "static/photos/{}/{}.jpg".format(unique_name,int(time.time()))
        fil = Image.open(file)
        fil.save(img_name)
        name_to_pt(file,unique_name)
        #cv2.imwrite(img_name,fil)
        #file.save(img_name)
        print(" saved: {}".format(img_name))
        flash("Uploaded successfully")
    return render_template("nrp.html")


def register_frame(text):
        fold = str(text)
        original_frame = cv2.imread("static/bin/clientimage.jpg")
        if not os.path.exists(os.path.join(IMAGE_FOLDER,fold)):
                    fold = uniquify_filename(fold)
                    os.mkdir(os.path.join(IMAGE_FOLDER,fold))
        #original_frame = frame.copy()
        mycol.insert_one(
                    {
                        'userfilename': 'webcam',
                        'unique_filename': fold,
                        'user_name':text,
                        'annotated': False
                    }
                    )
        # create directory if not exists
        img_name = "static/photos/{}/{}.jpg".format(fold, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        name_to_pt(img_name,fold)
        print(" saved: {}".format(img_name))

@app.route('/webcam_register',methods = ['GET','POST'])
def webcam_register():
    return render_template("canvas.html")

@app.route('/webcam_reg_pic',methods = ['GET','POST'])
def webcam_reg_pic():
    if request.method == "POST":
        image_data  = request.form.get("content").split(",")[1]
        text = request.form.get("name")
        if not text:
            return jsonify({'status': 'NG', "msg": "Registration failed Write your name correctly"})
            #flash("Error in Name , Write your name correctly")
            #return redirect((url_for(webcam_register)))
        with open("static/bin/clientimage.jpg" , "wb") as f:
            f.write(base64.b64decode(image_data))
        register_frame(text)
        return jsonify({'status': 'OK', "msg": "Registration successful"})
        #flash("SUCCESS : Your face is registered")
        #return redirect((url_for(webcam_register)))

@app.route('/')
def wrec():
    my_id = uuid.uuid1()
    session["user"] = my_id
    
    return render_template("wrec.html")

@app.route('/webcam_demo',methods = ["POST"])
def webcam_demo():
    #if request.method == "POST":
    img_encoded = request.form.get("content").split(",")[1]
    #print("Content: ", img_encoded[:30])
    img_decoded = base64.b64decode(img_encoded)
    #print("------",type(typ))
    imgdata = Image.open(io.BytesIO(img_decoded)).convert('RGB')
    # decoded = cv2.imdecode(np.frombuffer(typ, np.uint8), -1)
    if "user" in session:
        ses = str(session["user"])
        check_db(ses)
    else:
        ses = session['user'] = uuid.uuid1()
        ses = str(ses)
        check_db(ses)
        
    now = datetime.now()
    dt_string = str(now.strftime("%Y_%m_%d_%H%M%S%f"))
    save_path = 'static/session/{}/{}.jpg'.format(ses,dt_string)
    if not os.path.exists(os.path.join(session_fold,ses)):
        os.mkdir(os.path.join(session_fold,ses))
    imgdata.save(save_path)
    imgarray = np.array(imgdata)
    box_data = predict_frame(imgarray)
    #----------------from here we are storing boxdata
    cursor = framelist.find({'session':ses})
    all= cursor[0]
    framing = all["framing"]
    dic = all["out_names"]
    print("type-framing",type(framing))
    framing.append(box_data)
    if len(framing)>2 :
        framing.pop(0)
    dic = possibles_name(framing,dic)
    new_box_data = update_box_data(box_data,dic)
    framelist.replace_one({"session":ses} , {"session":ses , "framing" : framing,"out_names":dic})
    print("----framing :",framing)
    print("----dic :",dic)
    print("new_box_data",new_box_data)
    return jsonify(new_box_data)

def check_db(ses):
    count = framelist.count_documents({"session":ses})
    if count > 0:
        pass
    else:
        framelist.insert_one(
                                {"session":ses,
                                 "framing" : [],
                                 "out_names" : {}}
                            )

def update_box_data(box_data,dic):
    new_box_data = []
    for pro in box_data:
        if '{}'.format(pro["box"]) in dic:
            all_list = dic["{}".format(pro["box"])]
            name_list = all_list[0]
            # print(name_list)
            d = defaultdict(int)
            for i in name_list:
                d[i] += 1
            result = [max(d.items(), key=lambda x: x[1])]
            # print("---",result[0][0])
            # print("nameeeeeeee",pro["name"])
            if result[0][0] == pro["name"]:
                core= {}
                core["name"]=result[0][0]
                core["box"]=pro["box"]
                new_box_data.append(core)
            else:
                core = {}
                # core["name"]=" "
                core["box"]=pro["box"]
                new_box_data.append(core)

    return new_box_data

def possibles_name(framing,dic):
    for i in framing[0]:
        if "{}".format("name") in i:
            if "{}".format(i["box"]) not in dic:
                names = [[i["name"]],i["box"]]
                dic["{}".format(i["box"])]= names
                # print("new key added",dic)
            for h in framing[1]:
                if "{}".format("name") in h:
                    boxA = i["box"]
                    boxB = h["box"]
                    iou = similar_check(boxA,boxB)
                    # print("iou = ", iou)
                    if iou > 0.60:
                        if "{}".format(i['box']) in dic:
                            #print("=====" ,dic[i["name"]])
                            lis1 = dic["{}".format(i['box'])]
                            #print("------lis1",lis1)
                            if len(lis1[0])>= 5:
                                lis1[0].pop(0)
                            lis1[0].append(h["name"])
                            lis1[1]=boxB
                            dic.pop("{}".format(i["box"]))
                            dic["{}".format(h["box"])]= lis1
                            # print("__dic after append", dic)
                        else:
                            pass
                            # names = [[i["name"]],i["box"]]
                            # dic[i["name"]] = names
                            # print("dic key added before append", dic)
                    else:
                        pass
                        #dic["{}".format(h["name"])]= [h["name"]]
                        # print("after new adding new key",dic)
                else:
                    pass
        else:
            pass
    return dic

def similar_check(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #according to formula
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def predict_frame(frame):
    try:
        load_data = from_pt()
    except:
        save_data()
        load_data = from_pt()
    embedding_list = load_data[0]
    name_list = load_data[1]
    img = Image.fromarray(frame)
    #img.save("function.jpg")
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    box_data = []
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
        
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 

                dist_list = [] # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                fold = name_list[min_dist_idx] # get name corrosponding to minimum dist
                try:
                    cursor = mycol.count_documents({'unique_filename':fold})
                    #print("cursor-------------",cursor)
                except:
                    print("found nothing in monodb")
                    cursor = 0
                    pass
                if cursor > 0:
                    cursor = mycol.find({'unique_filename':fold })
                    imgrec = cursor[0]
                    #print(imgrec)
                    name = imgrec["user_name"]
                else:
                    name = ""
                box = boxes[i] 
                
                if min_dist<0.90:
                    box_data.append({"name":name , 'box': list(map(int, box[:4]))})
                #    frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)               
                #frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                else:
                    box_data.append({'box': list(map(int, box[:4]))})
    return box_data   

@app.route("/show_names")
def show_name():
    load_data = from_pt()
    name_list = load_data[1]
    #embedding_list = load_data[0]
    return "{}".format(name_list)

@app.route('/delete/<fold>')
def delete_from_pt(fold):
    load_data = from_pt()
    name_list = load_data[1]
    embedding_list = load_data[0]
    nameor = str(fold)
    i = 0
    Done = False
    for h in name_list:
        if h == nameor:
            Done = True
            name_list.pop(i)
            embedding_list.pop(i)
            #print(name_list)
            #print(embedding_list)
        i+=1
    if not Done:
        return "No name {} found in pt.".format(nameor)
    data = [embedding_list, name_list] 
    torch.save(data, pt_file)
    return "File name--- {} ---deleted from our data base.".format(fold)

# def generate_frames():
#     cam = cv2.VideoCapture(0)
#     while True: 
#         ## read the camera frame
#         success,frame=cam.read()
#         if not success:
#             break
#         else:
#             frame = predict_frame(frame)
#             ret,buffer=cv2.imencode('.jpg',frame)
#             frame=buffer.tobytes()
                        
#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/webcam_rec',methods = ["GET","POST"])
# def webcam_rec():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    
    app.run(debug=True, port=8085)
