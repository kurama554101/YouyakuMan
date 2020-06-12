import streamlit as st
import boto3
import json


def setup_ui():
    # setup sidebar
    endpoint_url = st.sidebar.text_input("input endpoint url of the inference server", "")
    region_name  = st.sidebar.text_input("input region name of the inference server", "us-east-2")
    input_mode = st.sidebar.selectbox("select the input mode", ("input", "file"))
    
    # setup main
    if input_mode == "file":
        setup_main_of_file_mode(endpoint_url, region_name)
    elif input_mode == "input":
        setup_main_of_hand_input_mode(endpoint_url, region_name)
    else:
        st.write("{} mode is not implemented!".format(input_mode))


def setup_main_of_file_mode(endpoint_url:str, region_name:str):
    # TODO : imp
    st.write("not implemented!")


def setup_main_of_hand_input_mode(endpoint_url:str, region_name:str):
    body = st.text_area(label="input_text", value="")
    start_btn = st.button("start to inference")
    if start_btn and len(body) > 0:
        with st.spinner("wait for inference..."):
            response = inference(endpoint_url, region_name, body)
        st.success("inference request is completed!!")
        
        st.write("Extractive summarization sentence : {}".format(response[0]))


def inference(endpoint_url:str, region_name:str, body:str):
    client = boto3.client("sagemaker-runtime", 
                          region_name=region_name,
                          )
    
    json_data = json.dumps(body)
    response = client.invoke_endpoint(
        EndpointName=endpoint_url,
        Body=json_data,
        ContentType="application/json",
        Accept="application/json"
    )
    
    response_body = response["Body"].read()
    response_body_json = json.loads(response_body)
    return response_body_json["result"]


def main():
    setup_ui()


if __name__ == "__main__":
    main()
