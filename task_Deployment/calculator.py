"""本地计算程序：负责执行四则运算。

Flask 应用在解析完请求后，会调用这里的 calculate() 完成计算，
从而把「Web 层」和「业务计算层」分离开来（对应题目中「调用本地的
另一个 py 程序计算」的要求）。
"""


import math


def calculate(operation, a, b):
    """对 a、b 执行 operation 指定的运算。

    operation 取值：
        sum      -> a + b
        multiply -> a * b
        subtract -> a - b
        divide   -> a / b

    抛出 ValueError：
        - operation 不在上述四种之中
        - divide 时除数为 0
    """
    try:
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (a, b)):
            raise ValueError('操作数必须是有限数字')
        if operation == 'sum':
            result = a + b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'divide':
            if b == 0:
                raise ValueError('除数不能为 0')
            result = a / b
        else:
            raise ValueError(f'不支持的 Operation：{operation!r}')
        if not math.isfinite(result):
            raise ValueError('结果超出有限数值范围')
        return result
    except OverflowError as error:
        raise ValueError('数字超出支持范围') from error
