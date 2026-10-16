"""Executa as células Python do notebook 02 sem servidor Jupyter.

Usa apenas a biblioteca padrão e as dependências de runtime. Preserva o 01.
Captura stdout e a representação textual da expressão final de cada célula.
"""
import ast
from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def execute():
    sys.path.insert(0,str(ROOT))
    path=ROOT/"notebooks/02_modelagem.ipynb"
    notebook=json.loads(path.read_text(encoding="utf-8"))
    scope={"__name__":"__notebook__"}
    count=0
    for cell in notebook["cells"]:
        if cell["cell_type"]!="code":
            continue
        count+=1
        source="".join(cell["source"])
        tree=ast.parse(source)
        expression=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
        captured=io.StringIO()
        with redirect_stdout(captured):
            exec(compile(tree,str(path),"exec"),scope)
            value=eval(compile(ast.Expression(expression.value),str(path),"eval"),scope) if expression else None
        outputs=[]
        if captured.getvalue():
            outputs.append({"output_type":"stream","name":"stdout","text":captured.getvalue().splitlines(keepends=True)})
        if value is not None:
            text=value.to_string() if hasattr(value,"to_string") else repr(value)
            outputs.append({"output_type":"execute_result","metadata":{},"execution_count":count,"data":{"text/plain":text.splitlines(keepends=True)}})
        cell["execution_count"]=count
        cell["outputs"]=outputs
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(f"Notebook 02: {count} células Python executadas; todas as verificações passaram.")


if __name__=="__main__":
    execute()
