#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# In[1]:


# %pip install streamlit
# %pip install face_recognition


# In[9]:


import face_recognition
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image
import pickle 
from os.path import exists, getmtime


# In[3]:


adhd_symptoms = ["miss details and are distracted easily",
"get bored quickly",
"have trouble focusing on a single task",
"have difficulty organizing thoughts and learning new information",
"lose pencils, papers, or other items needed to complete a task",
"don’t seem to listen",
"move slowly and appear as if they’re daydreaming",
"process information more slowly and less accurately than others",
"have trouble following directions",
"squirm, fidget, or feel restless",
"have difficulty sitting still",
"talk constantly",
"touch and play with objects, even when inappropriate to the task at hand",
"have trouble engaging in quiet activities",
"are constantly “on the go”",
"are impatient",
"act out of turn and don’t think about consequences of actions",
"blurt out answers and inappropriate comments"]


# In[4]:


def rando_result():
    
    adhd_status = np.random.randint(0, 2)
    adhd_symptom = ""
    if adhd_status == 1:
        adhd_symptom = np.random.choice(adhd_symptoms)
        
    return {
        "status": adhd_status, 
        "symptom": adhd_symptom
        }


# In[ ]:


seconds_in_a_week = 604800


# In[5]:


known_face_encodings = []
known_face_results = []

if exists('known_face_encodings.pkl'):   
    modified_time = getmtime('known_face_encodings.pkl')    
    age_of_modification_seconds = int(datetime.now().timestamp() - modified_time)
    if age_of_modification_seconds < seconds_in_a_week: # purge after 7 days of inactivity
        with open('known_face_encodings.pkl', 'rb') as f:
            known_face_encodings = pickle.load(f)    
            known_face_encodings = known_face_encodings[-1000:] # keep only the latest 1k embeddings in memory

if exists('known_face_results.pkl'):
    modified_time = getmtime('known_face_results.pkl')    
    age_of_modification_seconds = int(datetime.now().timestamp() - modified_time)
    if age_of_modification_seconds < seconds_in_a_week: # purge after 7 days of inactivity
        with open('known_face_results.pkl', 'rb') as f:
            known_face_results = pickle.load(f)
            known_face_results = known_face_results[-1000:] # keep only the latest 1k embeddings in memory


# In[ ]:


st.set_page_config(
    page_title="the thing", 
    page_icon="🪨", 
    layout="centered", 
    initial_sidebar_state="collapsed", 
    menu_items=None
    )


# In[ ]:


# with col_a:
head = st.markdown("### Do you have :blue[the thing]?")
sub_head = st.markdown("#### Scan your face to find out...")
picture = st.camera_input(label="Scan your face", label_visibility="collapsed")

# with col_b:
if picture:
    # st.image(picture)
    img = Image.open(picture)
    img_array = np.array(img)

    list_of_face_encodings = face_recognition.face_encodings(img_array)
    if len(list_of_face_encodings) > 0:
        face_encoding = list_of_face_encodings[0]
    
        results = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if len(known_face_encodings) == 0:
            known_face_encodings.append(face_encoding)
            known_face_results.append(rando_result())
            index_result = 0
        elif sum(results) == 0:
            # new face!
            known_face_encodings.append(face_encoding)
            known_face_results.append(rando_result())
            index_result = len(known_face_results) - 1
        else:
            # existing face
            index_result = np.where(results)[0][0]

        # print(known_face_results)
        if known_face_results[index_result]['status'] == 1:
            sub_head.markdown("#### :green[Yes.] You "+ known_face_results[index_result]['symptom'])
        else:
            sub_head.markdown("#### :red[No.] ")

    with open('known_face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)
    with open('known_face_results.pkl', 'wb') as f:
        pickle.dump(known_face_results, f)

st.markdown("No evilness, running this stupid [code](https://github.com/knhash/the-thing) for lolz.")
st.caption("Purging image data on server regularly (latest 1k images or 7 days of inactivity)")
    



# In[ ]:





# In[ ]:





# In[ ]:




