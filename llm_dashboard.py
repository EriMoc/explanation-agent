import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import httpx

from prompt_generator import generate_system_prompt
from utils.llm_utils import send_to_llm, exec_generated_code
from utils.analysis_utils import detect_anomalies, forecast, visualize_forecast, create_anomalies_dashboard_quarters
from utils.io_utils import save_to_excel

# 🔐 Načítanie .env súboru
load_dotenv()

# 🌐 LLM klient
client = OpenAI(
    base_url='https://genai-api-dev.dell.com/v1',
    http_client=httpx.Client(verify=False),
    api_key=os.environ["DEV_GENAI_API_KEY"]
)

# 🌟 UI rozhranie
st.set_page_config(page_title="LLM Dashboard", layout="wide")
st.title("📊 LLM Analytik pre Lokálne CSV")

uploaded_file = st.file_uploader("Nahraj CSV súbor", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV úspešne načítané.")
    st.dataframe(df.head())

    user_question = st.text_area("✍️ Tvoja otázka (napr. nájdi anomálie...)", height=100)

    if st.button("🔍 Analyzuj pomocou LLM"):
        with st.spinner("🧠 Generujem odpoveď..."):
            try:
                system_prompt = generate_system_prompt(df, df_name="df")
                generated_code = send_to_llm(client, system_prompt, user_question)

                st.subheader("🧾 Vygenerovaný kód:")
                st.code(generated_code, language="python")

                results = exec_generated_code(
                    generated_code,
                    dfs={"df": df},
                    df_var_name="df",
                    additional_funcs={
                        "detect_anomalies": detect_anomalies,
                        "forecast": forecast,
                        "save_to_excel": save_to_excel,
                        "visualize_forecast": visualize_forecast,
                    }
                )

                if results:
                    st.success("✅ Výsledky:")
                    for name, val in results.items():
                        if isinstance(val, pd.DataFrame):
                            st.write(f"**{name}**")
                            st.dataframe(val)

                            if "is_anomaly" in val.columns and "Period" in val.columns:
                                st.subheader("📈 Interaktívna vizualizácia anomálií")
                                fig = create_anomalies_dashboard_plotly(val, value_column="MKT Total Cost", period_column="Period")
                                st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("⚠️ Neboli vrátené žiadne výsledky.")

            except Exception as e:
                st.error(f"❌ Chyba: {str(e)}")
