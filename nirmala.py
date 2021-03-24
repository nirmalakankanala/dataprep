import streamlit as st
import pandas as pd
import numpy as np
import xlrd
import xlsxwriter
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import base64
from io import BytesIO
import cx_Oracle
import re
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from sqlalchemy import types, create_engine
import urllib.request




temp='\\temp.csv'

os.chdir(r'C:\Users\LENOVO\Desktop\heroku data preprocessing aplication') 


path=os.getcwd()

path=path+temp

path=(r"C:\Users\LENOVO\Desktop\heroku data preprocessing aplication\temp.csv")

st.title("Data Preprocessing Application")   

st.write("Hello")
st.sidebar.write("Data Import")

file_option=[".CSv",".Xlsx"]

file_select=st.sidebar.radio("please select a file type",file_option)


def upload_xlsx(uploaded_file):

    try:


        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            return df

    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

def upload_csv(uploaded_file):

    try:


        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            return df


    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



if file_select == ".CSv":
    file=st.file_uploader("please select file",type="csv")

    if file:

        if st.button('Upload File'):
            df=upload_csv(file)
            


if file_select == ".Xlsx":
    file=st.file_uploader("please select file",type="xlsx")

    if file:

        if st.button('Upload File'):
            df=upload_xlsx(file)
            
def main():
    try:
    
          way=["Missing value Treatment","Outlier Treatment","Scaling techniques"]
          T=st.sidebar.radio("please select a preprocessing method",way)
    
          if T=="Missing value Treatment":
        
              if T:
  
                 tech=["Mean","Median","Mode","Knn Method"]
                 U=st.sidebar.radio("Missing Value Treatment Methods ",tech)
                    
                 if U=="Mean":
                        if st.sidebar.button("process Mean"):
                            st.write('mean method')
                            df=pd.read_csv(path)
                            clean_df=(df.fillna(df.mean()))
                            df=clean_df
                            st.dataframe(clean_df)
                            clean_df.to_csv(path,index=False)
                            st.write('Data Types:',clean_df.dtypes)
                            st.write('Data description : ',clean_df.describe())
                            st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
                                
                  
                            
                 if U=="Median":
                       if st.sidebar.button("process Median"):
                            st.write('median method')
                            df=pd.read_csv(path)
                            clean_df=(df.fillna(df.median()))
                            df=clean_df
                            st.dataframe(clean_df)
                            clean_df.to_csv(path,index=False)
                            st.write('Data Types:',clean_df.dtypes)
                            st.write('Data description : ',clean_df.describe())
                            st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
                
                 if U=="Mode":
                        if st.sidebar.button("process Mode"):
                            st.write('Mode method')
                            df=pd.read_csv(path)
                            clean_df=(df.fillna(df.select_dtypes(include ='object').mode().iloc[0]))
                            df=clean_df
                            st.dataframe(clean_df)
                            clean_df.to_csv(path,index=False)
                            st.write('Data Types:',clean_df.dtypes)
                            st.write('Data description : ',clean_df.describe())
                            st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
                
                 if U=="Knn Method":
                        if st.sidebar.button("Knn Method"):
                            st.write('Knn Method')
                            df=pd.read_csv(path)
                            num_col =list(df.columns)
                            knn =KNNImputer(n_neighbors =1,add_indicator =True)
                            knn.fit(df[num_col])
                            knn_impute =pd.DataFrame(knn.transform(df[num_col]))
                            df[num_col]=knn_impute.iloc[:,:df[num_col].shape[1]]
                            clean_df= df
                            st.dataframe(clean_df)
                            clean_df.to_csv(path,index=False)
                            st.write('Data Types:',clean_df.dtypes)
                            st.write('Data description : ',clean_df.describe())
                            st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
                  
          if T=="Outlier Treatment":
              
              if T:
                    if st.sidebar.button("IQR"):
                        st.write('Inter Quantile Range')
                        df=pd.read_csv(path)
                        Q1=df.quantile(0.25)
                        Q3=df.quantile(0.75)
                        IQR=Q3-Q1
                        df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
                        st.dataframe(df)
                        st.write("Percentile Of Dataset :\n ", df.describe())
                        st.info('Size of dataset after outlier removal')
                        st.write(df.shape)
                        df.to_csv(path,index=False)
                    
          if T=="Scaling techniques":
            
              if T:
        
                       scaling=["Min Max Scalar","MaxAbs Scalar","Robust Scalar","Standard Scalar"]
                       W=st.sidebar.radio('scaling techniques',scaling)
                
                
                
                       if W=="Min Max Scalar":
                            if st.sidebar.button("Process MM"):
                                st.write('Min Max Scalar')
                                df=pd.read_csv(path)
                                x=df.select_dtypes(include=np.number)
                                min_x=np.min(x)
                                max_x=np.max(x)
                                xminmax = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
                                df=xminmax
                                st.dataframe(xminmax)
                                df.to_csv(path, index=False)
                    
                       if W=="MaxAbs Scalar":
                            if st.sidebar.button("Process MA"):
                                st.write('MaxAbs Scalar')
                                df=pd.read_csv(path)
                                x=df.select_dtypes(include=np.number)
                                max_abs_x=np.max(abs(x))
                                xmaxabs=x/np.max(abs(x))
                                st.dataframe(xmaxabs)
                                df=xmaxabs
                                df.to_csv(path, index=False)
                    
                       if W=="Robust Scalar":
                            if st.sidebar.button("Process RS"):
                                st.write('Robust Scalar')
                                df=pd.read_csv(path)
                                x=df.select_dtypes(include=np.number)
                                median_x=np.median(x)
                                q3=x.quantile(0.75)-x.quantile(0.25)
                                xrs=(x-np.median(x))/q3
                                st.dataframe(xrs)
                                df=xrs
                                df.to_csv(path, index=False)
                       if W=="Standard Scalar":
                            if st.sidebar.button("Process SS"):
                                st.write('Standard Scalar')
                                df=pd.read_csv(path)
                                X = df.select_dtypes(include=np.number)
                                mean_X = np.mean(X)
                                std_X = np.std(X)
                                Xstd = (X - np.mean(X))/np.std(X)
                                st.dataframe(Xstd)
                                df=Xstd
                                df.to_csv(path, index=False)
                    

    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
             
main()

def data_export():
    
    st.sidebar.write("Data Export")
    
    export_list = ["Xlsx","Csv"]
    
    export_select = st.sidebar.radio("Please select a export type",export_list)
    
    if export_select == "Csv":
        if st.sidebar.button("Download csv"):
            df=pd.read_csv(path)
            st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
            
    if export_select=="Xlsx":
        if st.sidebar.button("Download Xlsx"):
            df=pd.read_csv(path)
            st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
            
            
    


            
    
def get_table_download_link_csv(df):
    try:
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframes
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(
            csv.encode()
        ).decode()  # some strings <-> bytes conversions necessary here
        return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    

def to_excel(df):
    try:
        
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



def get_table_download_link_xlsx(df):
    try:
        
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        val = to_excel(df)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="dataprep.xlsx">Download xlsx file</a>' # decode b'abc' => abc
    

    
    
    
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
data_export()
