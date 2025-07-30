def send_to_llm(client, prompt, question, model="llama-3-3-70b-instruct"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().replace("```python", "").replace("```", "")

def exec_generated_code(code, dfs, df_var_name=None, additional_funcs={}):
    """
    Executes generated code with support for multiple DataFrames.
    dfs: dict of {df_var_name: DataFrame}
    """
    import pandas as pd

    # Prepare local variables: all DataFrames as variables
    local_vars = {name: df for name, df in dfs.items()}
    local_vars["pd"] = pd
    local_vars.update(additional_funcs)

    # Optionally replace old hardcoded names in code with actual variable names
    for name in dfs.keys():
        code = code.replace(name, name)

    exec(code, {},  local_vars)
    # Exclude DataFrames and additional functions from results
    exclude_keys = set(dfs.keys()) | set(additional_funcs.keys()) | {"pd"}
    results = {k: v for k, v in local_vars.items() if k not in exclude_keys}
    return results