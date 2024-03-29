from django.shortcuts import render
from .models import ChatRoom , Chat , User
from django.http import HttpResponse
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

# Create your views here.

def index(request):
    return render(request , 'index.html')

def home(request , receiver):
    try:
        User.objects.get(username = receiver)
    except:
        return HttpResponse("<h1>The User you want to chat with does'nt exist</h1>")
    if not request.user.is_authenticated:
        return HttpResponse("<h1>Please Login First</h1>")
    user_pair = [request.user.username , receiver]
    user_pair.sort()
    room_name = f'chat_{user_pair[0]}_{user_pair[1]}'
    # Get ChatRoom Object
    chat_room = ChatRoom.objects.filter(name=room_name).first()
    messages = []
    # if chatroom does'nt exist create one
    if not chat_room:
        chat_room = ChatRoom.objects.create(name=room_name)
        chat_room.members.add(request.user , User.objects.get(username=receiver))
    # retrieve all chat from Chat model where room is: chat_room in such a way so that we could distinguish between the messages of the sender and the receiver
    else:
        chat_messages =  Chat.objects.filter(room=chat_room)
        for message in chat_messages:
            if message.sender == request.user:
                messages.append({"text": message.message ,"timestamp": message.timestamp , "username": request.user.username}) # or message.sender.username
            elif message.receiver == request.user:
                messages.append({"text": message.message , "timestamp": message.timestamp , "username": message.sender.username})
    
    return render(request , 'home.html' ,{'receiver': receiver , 'messages': messages})

with open("app/model/toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open("app/model/severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open("app/model/obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open("app/model/insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open("app/model/threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open("app/model/identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open("app/model/toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open("app/model/severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open("app/model/obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open("app/model/insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open("app/model/threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open("app/model/identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)


def predict(request):
    if request.method == 'POST':
        user_input = request.POST.get('text')
        data = [user_input]

        vect = tox.transform(data)
        pred_tox = tox_model.predict_proba(vect)[:,1]

        vect = sev.transform(data)
        pred_sev = sev_model.predict_proba(vect)[:,1]

        vect = obs.transform(data)
        pred_obs = obs_model.predict_proba(vect)[:,1]

        vect = thr.transform(data)
        pred_thr = thr_model.predict_proba(vect)[:,1]

        vect = ins.transform(data)
        pred_ins = ins_model.predict_proba(vect)[:,1]

        vect = ide.transform(data)
        pred_ide = ide_model.predict_proba(vect)[:,1]

        out_tox = round(pred_tox[0], 2)
        out_sev = round(pred_sev[0], 2)
        out_obs = round(pred_obs[0], 2)
        out_ins = round(pred_ins[0], 2)
        out_thr = round(pred_thr[0], 2)
        out_ide = round(pred_ide[0], 2)

        print(out_tox)
        return render(request, 'index.html', {
                                                'pred_tox': 'Toxic: {}'.format(out_tox),
                                                'pred_sev': 'Severe Toxic: {}'.format(out_sev),
                                                'pred_obs': 'Obscene: {}'.format(out_obs),
                                                'pred_ins': 'Insult: {}'.format(out_ins),
                                                'pred_thr': 'Threat: {}'.format(out_thr),
                                                'pred_ide': 'Identity Hate: {}'.format(out_ide),
                                            })
    else:
        return render(request, 'index.html')
