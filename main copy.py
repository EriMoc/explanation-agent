import os
import httpx
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import IsolationForest

# from TEST import create_anomalies_dashboard_quarters
from prompt_generator import generate_system_prompt
from utils.io_utils import load_single_dataset, save_to_excel
from utils.analysis_utils import detect_anomalies, forecast, visualize_forecast, create_anomalies_dashboard_quarters
from utils.llm_utils import send_to_llm, exec_generated_code

# Load environment and suppress warnings
warnings.filterwarnings("ignore")
dotenv_path = "C:\\Users\\Erik_Mocik\\OneDrive - Dell Technologies\\Desktop\\DDC_API\\.env"
load_dotenv(dotenv_path, override=True)

# Load dataset
file_paths = [
    "inputs\\Total_Applicable_Cost_dataset.csv"
]
# 2. Load datasets into a dictionary
dfs = {}
for path in file_paths:
    df, dataset_name = load_single_dataset(path)
    df_var_name = os.path.splitext(os.path.basename(path))[0]
    dfs[df_var_name] = df
# 3. Generate system prompt (example: combine info from all datasets)
system_prompt = ""
for name, df in dfs.items():
    system_prompt += generate_system_prompt(df, name) + "\n"



# Setup LLM client
client = OpenAI(
    base_url='https://genai-api-dev.dell.com/v1',
    http_client=httpx.Client(verify=False),
    api_key=os.environ["DEV_GENAI_API_KEY"]
)

# User input
# user_question = "Find anomalies for mkt total cost in the dataset. Save it to excel and visualize."
user_question = "Forecast marketing external cost for the next 2 periods and visualize them"
# user_question = "Do you have internet access?"
# LLM generates code
generated_code = send_to_llm(client, system_prompt, user_question)
print("\nGenerated Code:\n", generated_code)

# Execute code
try:
    results = exec_generated_code(
        generated_code,
        dfs,
        df_var_name=None,
        additional_funcs={
            "detect_anomalies": detect_anomalies,
            "forecast": forecast,
            "save_to_excel": save_to_excel,
            "visualize_forecast": visualize_forecast,
            "create_anomalies_dashboard_quarters": create_anomalies_dashboard_quarters,
        }
    )

    if results:
        for name, value in results.items():
            print(f"\n✅ Result ({name}):\n", value)
    else:
        print("\n⚠️ No result returned.")

except Exception as e:
    print("\n❌ Error during execution:", str(e))
