# Langchain {#langchain}

## 案例 {#langchain_x}

### NL2SQL {#langchain_x_1}

如果可以的话，还应该加一个反馈机制：当第一次写出来的SQL语句在执行时发生错误后，不应该停止，而是将错误信息与错误的SQL语句喂给大模型，让大模型修正SQL语句，直至正确。


``` default
import os
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.agents.structured_output import ToolStrategy

# 大模型
llm = ChatOllama(
    model="qwen3:4b",
    base_url="http://localhost:11434",
    temperature=0
)

# 连接数据库
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")


DB_URI = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
)

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=["df_customers", "df_orders"]  # 白名单
)
table_info = db.get_table_info()   # 获取数据库中表结构

# 定义提示词模板
system_template = """
你是MySQL查询专家，严格遵守以下规则：
1. 仅执行查询操作，禁止增、删、改这些危险操作
2. 严格根据表结构进行查询，表结构信息：{table_info}
3. 调用工具前必须验证字段是否存在，生成SQL语句后自检语法
4. 当生成完SQL语句后，需要执行SQL语句，从数据库中提取相应的数据
5. 如果涉及多表连接、聚合等复杂操作，必要时可以创建临时表、使用子查询等方法
"""
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
user_prompt = HumanMessagePromptTemplate.from_template("{question}")
prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    user_prompt
])

# 定义工具
def generate_sql(question: str) -> str:
    """根据自然语言问题生成一条 MySQL SQL语句。"""
    msg = prompt.format_messages(
        table_info = table_info,
        question=question
    )
    sql = llm.invoke(msg).content.strip()

    return sql

def run_sql(sql: str) -> dict:
    """在数据库中执行SQL语句，提取相应的数据"""
    with db._engine.connect() as con:
        df = pd.read_sql(sql, con)
        df = df.to_dict(orient="records")
        return df

# 格式化输出
@dataclass
class FinalAnswer:
    """输出结果"""
    sql: str
    df: List[Dict[str, Any]]
    explanation: str

# 定义智能体
agent = create_agent(
    model=llm,
    tools=[generate_sql, run_sql],
    response_format=ToolStrategy(FinalAnswer)
)

def sql_query_agent(question):
    out = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    sql = out["structured_response"].sql
    df = pd.DataFrame(out["structured_response"].df)
    explanation = out["structured_response"].explanation
    
    return sql, df, explanation

# 流式输出
# for step in agent.stream(
#     {"messages": [{"role": "user", "content": "查询ID为2的用户的个人信息及其总下单次数，总下单次数列用“order_num”表示"}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

if __name__ == "__main__":
    question = "查询ID为2的用户的个人信息及其总下单次数，总下单次数列用“order_num”表示"
    sql, df, _ = sql_query_agent(question)
```

