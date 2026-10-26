import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

st.set_page_config(page_title="Hospital Appointment Intelligence", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#090f1c;color:#e5e7eb}
[data-testid="stMetric"]{background:#162238;border:1px solid #263650;border-radius:14px;padding:16px}
h1,h2,h3{color:#e5e7eb}
.block-container{padding-top:1.8rem;max-width:1450px}
</style>""",unsafe_allow_html=True)
BG="#0b1220"; BLUE="#38bdf8"; ORANGE="#fb923c"; PURPLE="#a78bfa"; GREEN="#34d399"
REQUIRED=['AppointmentID','Age','Gender','ScheduledDay','AppointmentDay','No-show','SMS_received','Hipertension','Diabetes','Alcoholism','Handcap','Scholarship']
FEATURES=['Age','Gender','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','WaitingDays','Weekday']

@st.cache_data(show_spinner="Cleaning appointment records...")
def load_data(raw):
    from io import BytesIO
    df=pd.read_csv(BytesIO(raw))
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError("Missing dataset columns: "+", ".join(missing))
    df['Age']=pd.to_numeric(df['Age'],errors='coerce')
    d=df.drop_duplicates(subset='AppointmentID').dropna(subset=REQUIRED).copy()
    d=d[d.Age.between(0,115)&d['No-show'].isin(['Yes','No'])&d.Gender.isin(['F','M'])].copy()
    for c in ['ScheduledDay','AppointmentDay']:
        d[c]=pd.to_datetime(d[c],utc=True,errors='coerce')
    d=d.dropna(subset=['ScheduledDay','AppointmentDay'])
    d['WaitingDays']=(d.AppointmentDay.dt.normalize()-d.ScheduledDay.dt.normalize()).dt.days
    d=d[d.WaitingDays>=0].copy()
    d['NoShow']=d['No-show'].eq('Yes').astype(int)
    d['AgeGroup']=pd.cut(d.Age,[-1,18,35,60,115],labels=['0-18','19-35','36-60','Above 60'])
    d['Weekday']=d.AppointmentDay.dt.day_name()
    d['AppointmentMonth']=d.AppointmentDay.dt.strftime('%Y-%m')
    d['SMS Label']=d.SMS_received.map({0:'No SMS',1:'SMS Received'})
    d['WaitingGroup']=pd.cut(d.WaitingDays,[-1,0,3,7,14,30,np.inf],labels=['Same day','1–3','4–7','8–14','15–30','31+'])
    return d,len(df)

@st.cache_resource(show_spinner="Training Decision Tree...")
def train_model(raw):
    d,_=load_data(raw)
    X=d[FEATURES];y=d.NoShow
    numeric=[c for c in FEATURES if c not in ['Gender','Weekday']]
    pre=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore'),['Gender','Weekday']),('numbers','passthrough',numeric)])
    model=Pipeline([('preprocess',pre),('classifier',DecisionTreeClassifier(max_depth=8,min_samples_leaf=50,class_weight='balanced',random_state=42))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    metrics={'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1 Score':f1_score(yte,pred,zero_division=0)}
    cm=confusion_matrix(yte,pred,labels=[0,1])
    names=model.named_steps['preprocess'].get_feature_names_out()
    importance=pd.DataFrame({'Feature':names,'Importance':model.named_steps['classifier'].feature_importances_}).nlargest(12,'Importance')
    return model,metrics,cm,importance

def style(fig,height=390):
    fig.update_layout(template='plotly_dark',paper_bgcolor=BG,plot_bgcolor=BG,font_color='#e5e7eb',height=height,margin=dict(l=35,r=20,t=55,b=35),showlegend=False)
    fig.update_yaxes(gridcolor='#25334b',zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig

st.title("🏥 Hospital Appointment Intelligence")
st.caption("PADV Activity 2 | Healthcare — Predicting Hospital Appointment No-Shows | Python + Streamlit + Decision Tree")
default=Path(__file__).with_name('KaggleV2-May-2016.csv')
with st.sidebar:
    st.header("Data source")
    uploaded=st.file_uploader("Upload KaggleV2-May-2016.csv",type=['csv'])
    if uploaded is None and default.exists():
        raw=default.read_bytes()
        st.success("Using CSV bundled with app")
    elif uploaded is not None:
        raw=uploaded.getvalue()
    else:
        raw=None
        st.info("Upload the original Kaggle CSV to start. If deploying publicly, place the CSV beside app.py to show the dashboard immediately.")
if raw is None:
    st.markdown("### Upload the hospital appointments dataset to view the live dashboard.")
    st.markdown("Dataset: https://www.kaggle.com/datasets/joniarroba/noshowappointments")
    st.stop()
try:
    d,original=load_data(raw)
    if len(d)<20 or d.NoShow.nunique()<2: st.error("Insufficient data or missing attendance classes.");st.stop()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()
with st.sidebar:
    st.divider()
    st.header("Interactive filters")
    age=st.multiselect("Age group",['0-18','19-35','36-60','Above 60'],default=['0-18','19-35','36-60','Above 60'])
    gender=st.multiselect("Gender",['F','M'],default=['F','M'])
    sms=st.selectbox("SMS reminder",['All','Received','Not received'])
    condition=st.selectbox("Medical condition",['All','Diabetes','Hypertension','Alcoholism'])
    st.caption("Filters affect dashboard analytics. Model training always uses the full cleaned dataset.")
f=d[d.AgeGroup.astype(str).isin(age)&d.Gender.isin(gender)].copy()
if sms!='All':f=f[f.SMS_received.eq(1 if sms=='Received' else 0)]
if condition!='All':f=f[f[{'Diabetes':'Diabetes','Hypertension':'Hipertension','Alcoholism':'Alcoholism'}[condition]].eq(1)]
total=len(f);missed=int(f.NoShow.sum());attended=total-missed
a,b,c,e=st.columns(4)
a.metric("Appointments",f"{total:,}")
b.metric("Attended",f"{attended:,}")
c.metric("Missed",f"{missed:,}")
e.metric("No-show rate",f"{missed/total:.2%}" if total else "N/A")
overview,patterns,model_tab,simulator,downloads=st.tabs(["📊 Overview","🔎 Patient Analytics","🧠 Model Performance","🎯 Risk Simulator","⬇️ Downloads"])
def rate(col,order=None):
    s=f.groupby(col,observed=True).NoShow.agg(['mean','count']).reset_index()
    if order is not None:
        s[col]=pd.Categorical(s[col],categories=order,ordered=True)
        s=s.sort_values(col)
    s['Rate']=s['mean']*100
    return s
with overview:
    if f.empty:st.warning("No appointments match these filters.")
    else:
        left,right=st.columns(2)
        with left:
            fig=go.Figure(go.Pie(labels=['Attended','Missed'],values=[attended,missed],hole=.65,marker_colors=[BLUE,ORANGE],textinfo='label+percent'))
            fig.update_layout(title='Attendance Breakdown')
            st.plotly_chart(style(fig),use_container_width=True)
        with right:
            s=rate('AgeGroup',['0-18','19-35','36-60','Above 60'])
            fig=px.bar(s,x='AgeGroup',y='Rate',color_discrete_sequence=[PURPLE],text='Rate',title='Age Group No-Show Rate (%)')
            fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
            st.plotly_chart(style(fig),use_container_width=True)
        s=f.groupby('AppointmentMonth').agg(Appointments=('NoShow','size'),Missed=('NoShow','sum')).reset_index()
        fig=px.line(s,x='AppointmentMonth',y=['Appointments','Missed'],markers=True,title='Appointment Volume and Missed Appointments by Month')
        fig.update_layout(showlegend=True,legend_title_text='')
        st.plotly_chart(style(fig,350),use_container_width=True)
with patterns:
    if f.empty:st.warning("No matching appointments.")
    else:
        l,r=st.columns(2)
        with l:
            s=rate('WaitingGroup',['Same day','1–3','4–7','8–14','15–30','31+'])
            fig=px.bar(s,x='WaitingGroup',y='Rate',color_discrete_sequence=[BLUE],title='No-Show Rate by Waiting Time')
            st.plotly_chart(style(fig),use_container_width=True)
        with r:
            cond=[]
            for name,col in [('Diabetes','Diabetes'),('Hypertension','Hipertension'),('Alcoholism','Alcoholism')]:
                sub=f[f[col].eq(1)]
                cond.append({'Condition':name,'Rate':sub.NoShow.mean()*100 if len(sub) else np.nan,'Appointments':len(sub)})
            fig=px.bar(pd.DataFrame(cond),x='Condition',y='Rate',color_discrete_sequence=[ORANGE],title='No-Show Rate Among Patients with Each Condition')
            st.plotly_chart(style(fig),use_container_width=True)
        l,r=st.columns(2)
        with l:
            s=rate('Weekday',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            fig=px.bar(s,x='Weekday',y='Rate',color_discrete_sequence=[GREEN],title='No-Show Rate by Appointment Weekday')
            st.plotly_chart(style(fig),use_container_width=True)
        with r:
            s=rate('SMS Label',['No SMS','SMS Received'])
            fig=px.bar(s,x='SMS Label',y='Rate',color_discrete_sequence=[PURPLE],title='SMS Reminder Comparison')
            st.plotly_chart(style(fig),use_container_width=True)
        st.info("These are observational associations. SMS reminders and other variables should not be interpreted as causing no-shows.")
with model_tab:
    model,metrics,cm,importance=train_model(raw)
    st.subheader("Decision Tree classification")
    st.caption("80/20 stratified train-test split • max_depth=8 • min_samples_leaf=50 • class_weight='balanced' • random_state=42")
    cols=st.columns(4)
    for col,(name,val) in zip(cols,metrics.items()):col.metric(name,f"{val:.3f}")
    l,r=st.columns(2)
    with l:
        fig=px.imshow(cm,text_auto=True,x=['Attended','Missed'],y=['Attended','Missed'],labels={'x':'Predicted','y':'Actual'},color_continuous_scale='Blues',title='Confusion Matrix')
        st.plotly_chart(style(fig,430),use_container_width=True)
    with r:
        fig=px.bar(importance.sort_values('Importance'),x='Importance',y='Feature',orientation='h',title='Top Prediction Features',color_discrete_sequence=[BLUE])
        st.plotly_chart(style(fig,430),use_container_width=True)
    st.warning("Accuracy alone can be misleading on this imbalanced dataset. Check recall and precision for missed appointments.")
with simulator:
    model,_,_,_=train_model(raw)
    st.subheader("Single Appointment Risk Simulator")
    st.caption("Illustrative model prediction, not a clinically validated risk score.")
    with st.form('predict'):
        l,r=st.columns(2)
        with l:
            age_input=st.slider('Age',0,110,30)
            wait=st.slider('Waiting days',0,120,7)
            gender_input=st.selectbox('Gender',['F','M'])
            sms_input=st.selectbox('SMS received',[0,1],format_func=lambda v:'Yes' if v else 'No')
        with r:
            weekday=st.selectbox('Appointment weekday',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
            diabetes=st.checkbox('Diabetes')
            hyper=st.checkbox('Hypertension')
        submitted=st.form_submit_button('Predict no-show risk',type='primary')
    if submitted:
        row=pd.DataFrame([{'Age':age_input,'Gender':gender_input,'Scholarship':0,'Hipertension':int(hyper),'Diabetes':int(diabetes),'Alcoholism':0,'Handcap':0,'SMS_received':sms_input,'WaitingDays':wait,'Weekday':weekday}])
        probability=float(model.predict_proba(row)[0,1])
        prediction=int(model.predict(row)[0])
        st.metric('Estimated no-show probability',f'{probability:.1%}')
        st.success('Predicted: MISSED' if prediction else 'Predicted: ATTENDED')
        st.caption("This score is based on the fitted model and is not a calibrated medical probability.")
with downloads:
    st.subheader("Download analysis outputs")
    st.download_button("Cleaned hospital appointments (CSV)",d.to_csv(index=False).encode(),file_name="Cleaned_Hospital_Appointments.csv",mime='text/csv')
    _,metrics,_,_=train_model(raw)
    st.download_button("Model evaluation metrics (CSV)",pd.DataFrame([metrics]).to_csv(index=False).encode(),file_name="Model_Metrics.csv",mime='text/csv')
    st.write(f"Original rows: {original:,} · Cleaned rows: {len(d):,} · Removed: {original-len(d):,}")
    st.markdown("Original dataset: [Medical Appointment No Shows on Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)")
