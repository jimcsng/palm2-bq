import sqlparse
import traceback
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import streamlit.components.v1 as components

from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain import LLMChain
from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts.prompt import PromptTemplate

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
import json
from PIL import Image
from pathlib import Path
import base64

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

table_imgs = {
    "HKEX_Securities_Trades_Table": "trade_record.png",
    "HKEX_Securities_Orders_Table": "order.png",
    "Reference_Price_Table": "ref_price.png"
}

print('Initializing VertexAI...')
aiplatform.init(project="cloud-llm-preview1", location="us-central1")
llm = VertexAI(model_name="code-bison", max_output_tokens=1024)

print('Initializing BigQuery...')
project_id = 'sunivy-for-example'
# dataset_id = 'blackbelt_capstone_healthcare'
# dataset_id = 'jimng'
# table_uri = f"bigquery://{project_id}/{dataset_id}"
# engine = create_engine(f"bigquery://{project_id}/{dataset_id}")

dataset_id = 'HKEx_demo_securities_market'
table_uri = f"bigquery://{project_id}/{dataset_id}"
engine = create_engine(f"bigquery://{project_id}/{dataset_id}")

# Example Query
# example:
# Asked: What is the moving average, windowed in hour, of the equity traded for 336 in 2021-05-15?
# You should reply: SELECT TIMESTAMP_TRUNC(datetime, HOUR), AVG(price) AS moving_average FROM `sunivy-for-example.HKEx_demo_securities_market.HKEX_Securities_Trades_Table` WHERE security_code = 336 AND datetime BETWEEN '2021-05-15 00:00:00' AND '2021-05-15 23:00:00' GROUP BY TIMESTAMP_TRUNC(datetime, HOUR)
# when asked for equity, it means the rows with column instrument_type being EQTY.
# when asked for the security code of a stock, please use string similarty lookup instead of using exact match.

# The prompt for generating sql from the question asked
_googlesql_prompt = """You are a GoogleSQL and a stock market data expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table. 
When asked for moving averages, please use the timestamp_trunc function to extract the window accordingly. Also for the answer will respond in a full human readable sentence.

for numbers in the answer, please round it to the nearest 3 decimal points.

example:
If Asked: What is the moving average, windowed in hour, of the equity traded for 336 in 2021-05-15?
You should reply: SELECT TIMESTAMP_TRUNC(datetime, HOUR), AVG(price) AS moving_average FROM `sunivy-for-example.HKEx_demo_securities_market.HKEX_Securities_Trades_Table` WHERE SecurityCode = 336 AND datetime BETWEEN '2021-05-15 00:00:00' AND '2021-05-15 23:00:00' GROUP BY TIMESTAMP_TRUNC(datetime, HOUR)


Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
{table_info}

Question: {input}"""

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_googlesql_prompt,
)
# final_prompt = GOOGLESQL_PROMPT.format(input='What is the security code of the most traded equity on 2021-05-15 entire day?', table_info='HKEX_Securities_Orders, HKEX_Securities_Ref_Price, HKEX_Securities_TradeEvents', top_k=2000000)

# prompt for synthesizing the answer from the question, sql and the sql result
_result_synthesis_prompt = """Inspect the question asked: {question}
The SQL query asked is {sql}
The SQL result is {sql_result}

Please summarize and provide a human friend answer to the original question.
"""

SYNTHESIS_PROMPT = PromptTemplate(
    input_variables=["question", "sql", "sql_result"],
    template=_result_synthesis_prompt,
)

# prompt for extracting the table used in the sql and show the table details page url
_get_table_used_prompt = """From the sql {sql}, first extract the table name that the sql is querying against
then provide the result in the a valid json with the following format:
{{"table_names": [<the table name you have extracted>]}}
"""

GET_TABLE_PROMPT = PromptTemplate(
    input_variables=["sql"],
    template=_get_table_used_prompt,
)

db = SQLDatabase(engine=engine,metadata=MetaData(bind=engine),include_tables=['HKEX_Securities_Orders_Table', 'HKEX_Securities_Trades_Table', 'Reference_Price_Table'])
# db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=GOOGLESQL_PROMPT, verbose=False, return_intermediate_steps=True)
db_chain = create_sql_query_chain(llm,db,prompt=GOOGLESQL_PROMPT, k=20)
answer_chain = LLMChain(llm=llm, prompt=SYNTHESIS_PROMPT)
get_table_chain = LLMChain(llm=llm, prompt=GET_TABLE_PROMPT)

def clean_sql_gen_result(result):
    return result.replace("```sql", "").replace("```", "")

def clean_get_table_result(result):
    return result.replace("```json", "").replace("```", "")

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("header.png")
)
hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
</style>

"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# st.markdown(
#     header_html, unsafe_allow_html=True,
# )
# components.html(header_html)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)
image = Image.open("header.png")
st.image(image, width=streamlit_js_eval(js_expressions='screen.width', key = 'SCR'))

text = "What is the most traded equity on 2021-05-15?"
question = st.text_area("Question", placeholder=text)

if st.button("Send") or question:
    try:
        with st.spinner('Wait for it...'):
            result = db_chain.invoke({"question":question})
            sql = clean_sql_gen_result(result)
            st.markdown("## Query")
            st.code(sqlparse.format(sql, reindent=True, keyword_case='upper'), language='sql')

            sql_result = db.run(sql)
            answer = answer_chain.predict(question=question, sql_result=sql_result, sql=sql)
            st.markdown("## Answer")
            st.markdown(answer)

            st.markdown("## Chart (if plottable)")
            query_result = engine.execute(sql)
            result_df = pd.DataFrame([dict(i) for i in query_result])
            result_df['moving_average'] = result_df['moving_average'].astype(float)
            result_df = result_df.rename(columns={'f0_':'datetime'})
            st.line_chart(result_df, x='datetime', y='moving_average')

            table_result = get_table_chain.predict(sql=sql)
            table_result = clean_get_table_result(table_result)
            table_names = json.loads(table_result)["table_names"]
            print(table_names)

            table_imgs_list = list(map(lambda table: {"img": table_imgs[table], "caption": table}, table_names))
            print(table_imgs_list)

            st.markdown("## Table(s) Used")
            for img in table_imgs_list:
                image = Image.open(img['img'])
                st.image(image, caption="")
            

            # result = db_chain(question)
            # sql_generated = result["intermediate_steps"][1]
            # sql_result = result["intermediate_steps"][3]
            # sql_answer = result["intermediate_steps"][5]

            # print(sqlparse.format(sql_generated, reindent=True, keyword_case='upper'))
            # print(sql_result)
            # st.markdown("## Query")
            # st.code(sqlparse.format(sql_generated, reindent=True, keyword_case='upper'), language='sql')
            # st.markdown("## Answer")
            # st.markdown(sql_answer)
            # st.markdown("## Chart (if plottable)")
            # result_df['moving_average'] = result_df['moving_average'].astype(float)
            # result_df = result_df.rename(columns={'f0_':'datetime'})
            # st.line_chart(result_df, x='datetime', y='moving_average')
    except Exception as e:
        # print error and traceback
        print(f"error: {e}")
        print(traceback.format_exc())
        # st.error(e)
